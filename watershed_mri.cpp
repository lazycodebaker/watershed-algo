/**
 * @file watershed_mri_segmentation.cpp
 * @brief Demonstrates tumor/organ segmentation in MRI images using the Watershed algorithm with OpenCV.
 * @author [Your Name/Organization]
 * @date [Current Date]
 *
 * This program loads an MRI image, preprocesses it, generates markers for the
 * Watershed algorithm, applies the segmentation, and visualizes the results.
 * Intermediate steps can be displayed in a grid format for better understanding.
 * The final segmented output is also displayed in a grid and saved to disk.
 */

#include <opencv2/opencv.hpp> // Core OpenCV functionalities and image processing
#include <iostream>            // For standard I/O (cin, cout, cerr)
#include <vector>              // For std::vector
#include <string>              // For std::string
#include <algorithm>           // For std::min, std::max_element, std::minmax_element
#include <random>              // For std::mt19937, std::uniform_int_distribution (C++11 random numbers)
#include <cmath>               // For std::ceil
#include <set>                 // For std::set to get unique labels
#include <map>                 // For std::map to store colors for labels

// --- Configuration Parameters ---

/**
 * @struct Config
 * @brief Holds all tunable parameters for the MRI segmentation pipeline.
 *
 * This structure centralizes configuration values, making it easier to adjust
 * the behavior of the algorithm without modifying the core logic.
 */
struct Config
{
    // Preprocessing parameters
    cv::Size gaussian_blur_kernel_size = cv::Size(5, 5); ///< Kernel size for Gaussian blur noise reduction.
    int adaptive_thresh_block_size = 11;                 ///< Block size for adaptive thresholding (must be odd).
    double adaptive_thresh_c = 2.0;                      ///< Constant subtracted from the mean in adaptive thresholding.

    // Morphological operation parameters for marker generation
    cv::Size morph_open_kernel_size = cv::Size(3, 3);    ///< Kernel size for morphological opening (noise removal).
    cv::Size morph_dilate_kernel_size = cv::Size(5, 5);  ///< Kernel size for dilation to find sure background.
    double dist_transform_threshold_ratio = 0.4;         ///< Ratio of max distance for thresholding distance transform to find sure foreground.
    cv::Size sure_fg_erode_kernel_size = cv::Size(3, 3); ///< Kernel size for eroding sure foreground markers for refinement.

    // Display and output settings
    bool display_intermediate_steps = true;              ///< Flag to control display of intermediate processing steps.
    int grid_num_cols = 3;                               ///< Number of columns for displaying images in a grid.
    cv::Size grid_cell_image_size = cv::Size(300, 250);  ///< Target size for each image within a grid cell.
    int grid_title_height = 30;                          ///< Height reserved for the title above each image in the grid.
    int grid_padding = 10;                               ///< Padding around grid cells and the grid itself.
    cv::Scalar grid_background_color = cv::Scalar(255, 255, 255);      ///< Background color for the image grid (BGR: White).
    cv::Scalar grid_cell_background_color = cv::Scalar(240, 240, 240); ///< Background color for individual cells in the grid (BGR: Light Gray).
    std::string output_grid_filename_prefix = "mri_segmentation_summary"; ///< Prefix for the saved output grid image filename.
};

// --- Helper Functions ---

/**
 * @brief Prepares an input image for display in a grid by converting it to CV_8UC3 format.
 * @param input_img The input cv::Mat image, which can be of various types (e.g., grayscale, float, int).
 * @return cv::Mat A 3-channel, 8-bit unsigned char image (CV_8UC3) suitable for display.
 *                 Handles empty images, grayscale, float, and integer marker images.
 */
cv::Mat prepareImageForGrid(const cv::Mat &input_img)
{
    cv::Mat display_ready_img;

    // Handle empty input image
    if (input_img.empty())
    {
        display_ready_img = cv::Mat(cv::Size(100, 100), CV_8UC3, cv::Scalar(50, 50, 50)); // Dark gray placeholder
        cv::putText(display_ready_img, "Empty", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        return display_ready_img;
    }

    // If already in desired format (CV_8UC3)
    if (input_img.channels() == 3 && input_img.type() == CV_8UC3)
    {
        display_ready_img = input_img.clone();
    }
    // Handle single-channel images
    else if (input_img.channels() == 1)
    {
        if (input_img.type() == CV_8U) // Grayscale 8-bit
        {
            cv::cvtColor(input_img, display_ready_img, cv::COLOR_GRAY2BGR);
        }
        else if (input_img.type() == CV_32F || input_img.type() == CV_64F) // Float images (e.g., distance transform)
        {
            cv::Mat normalized_img;
            cv::normalize(input_img, normalized_img, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::cvtColor(normalized_img, display_ready_img, cv::COLOR_GRAY2BGR);
        }
        else if (input_img.type() == CV_32S) // Integer images (e.g., raw markers before colormap)
        {
            // This case is best handled by `applyColormapToMarkers` before calling this function.
            // However, as a fallback, attempt a basic colormap application.
            cv::Mat temp_8u;
            double minVal, maxVal;
            cv::minMaxLoc(input_img, &minVal, &maxVal);
            if (minVal == maxVal) // Uniform image
                input_img.convertTo(temp_8u, CV_8U, 1.0, -minVal); // Map minVal to 0
            else // Normalize and scale
                input_img.convertTo(temp_8u, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
            cv::applyColorMap(temp_8u, display_ready_img, cv::COLORMAP_JET);
        }
        else // Fallback for other single-channel types
        {
            cv::Mat temp_8u;
            input_img.convertTo(temp_8u, CV_8U); // Attempt conversion to 8-bit unsigned
            cv::cvtColor(temp_8u, display_ready_img, cv::COLOR_GRAY2BGR);
        }
    }
    // Fallback for other multi-channel images (e.g., CV_32FC3)
    else
    {
        cv::Mat temp_8uc3;
        input_img.convertTo(temp_8uc3, CV_8UC3, 255.0); // Basic scaling (might not be ideal for all cases)
        display_ready_img = temp_8uc3;
    }
    return display_ready_img;
}

/**
 * @brief Creates a composite image grid from a vector of images and their titles.
 * @param images_with_titles A vector of pairs, where each pair contains a cv::Mat image and its std::string title.
 * @param cfg The global Config struct containing grid layout parameters.
 * @param force_num_cols Optional. If > 0, overrides the number of columns specified in cfg.
 * @return cv::Mat A single image containing all input images arranged in a grid with titles.
 *                 Returns an empty cv::Mat if the input vector is empty.
 */
cv::Mat createImageGrid(const std::vector<std::pair<cv::Mat, std::string>> &images_with_titles, const Config &cfg, int force_num_cols = -1)
{
    if (images_with_titles.empty())
    {
        return cv::Mat(); // Return empty Mat if no images to display
    }

    int num_images = static_cast<int>(images_with_titles.size());
    int num_cols = (force_num_cols > 0) ? force_num_cols : cfg.grid_num_cols;
    // Ensure num_cols does not exceed the number of images
    if (num_images < num_cols)
        num_cols = num_images;

    int num_rows = static_cast<int>(std::ceil(static_cast<double>(num_images) / num_cols));

    // Calculate dimensions for each cell (image + title space)
    int cell_width = cfg.grid_cell_image_size.width;
    int cell_height = cfg.grid_cell_image_size.height + cfg.grid_title_height;

    // Calculate total grid dimensions including padding
    int grid_width = num_cols * (cell_width + cfg.grid_padding) + cfg.grid_padding;
    int grid_height = num_rows * (cell_height + cfg.grid_padding) + cfg.grid_padding;

    cv::Mat grid_image(grid_height, grid_width, CV_8UC3, cfg.grid_background_color);

    // Iterate through images and place them into the grid
    for (int i = 0; i < num_images; ++i)
    {
        int row_idx = i / num_cols;
        int col_idx = i % num_cols;

        // Prepare the source image (convert type, resize)
        cv::Mat source_img_prepared = prepareImageForGrid(images_with_titles[i].first);
        const std::string &title = images_with_titles[i].second;

        cv::Mat resized_img;
        cv::resize(source_img_prepared, resized_img, cfg.grid_cell_image_size, 0, 0, cv::INTER_AREA);

        // Create an individual cell Mat
        cv::Mat cell_mat(cell_height, cell_width, CV_8UC3, cfg.grid_cell_background_color);

        // Add title text to the cell
        cv::putText(cell_mat, title, cv::Point(5, cfg.grid_title_height - 10), // Position text
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        // Define ROI for the image within the cell (below the title)
        cv::Rect image_roi(0, cfg.grid_title_height, cfg.grid_cell_image_size.width, cfg.grid_cell_image_size.height);
        resized_img.copyTo(cell_mat(image_roi));

        // Define ROI for the cell within the main grid
        int offset_x = cfg.grid_padding + col_idx * (cell_width + cfg.grid_padding);
        int offset_y = cfg.grid_padding + row_idx * (cell_height + cfg.grid_padding);
        cv::Rect grid_roi(offset_x, offset_y, cell_width, cell_height);
        cell_mat.copyTo(grid_image(grid_roi));
    }
    return grid_image;
}

/**
 * @brief Applies a JET colormap to a single-channel integer matrix (typically markers).
 * @param markers_int32 Input cv::Mat, expected to be of type CV_32S.
 * @return cv::Mat A CV_8UC3 image with the JET colormap applied.
 *                 Returns a placeholder error image if input is invalid.
 */
cv::Mat applyColormapToMarkers(const cv::Mat &markers_int32)
{
    // Validate input
    if (markers_int32.empty() || markers_int32.type() != CV_32S)
    {
        cv::Mat error_img(cv::Size(100, 100), CV_8UC3, cv::Scalar(0, 0, 255)); // Red placeholder
        cv::putText(error_img, "Invalid Markers", cv::Point(5, 50), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        return error_img;
    }

    cv::Mat markers_8u, colored_markers;
    double minVal, maxVal;
    cv::minMaxLoc(markers_int32, &minVal, &maxVal); // Find min and max label values

    // Normalize the marker image to 0-255 range for colormap application
    if (minVal == maxVal) // Handle uniform image (e.g., all zeros or all same label)
    {
        // Map the single value to 0 to ensure it's visible in the colormap (often as blue in JET)
        markers_int32.convertTo(markers_8u, CV_8U, 1.0, -minVal);
    }
    else
    {
        cv::normalize(markers_int32, markers_8u, 0, 255, cv::NORM_MINMAX, CV_8U);
    }
    cv::applyColorMap(markers_8u, colored_markers, cv::COLORMAP_JET);
    return colored_markers;
}

// --- Core Processing Functions ---

/**
 * @brief Loads, converts to grayscale, and preprocesses an MRI image.
 * @param image_path Path to the input MRI image file.
 * @param cfg The global Config struct.
 * @param[out] out_enhanced_img The preprocessed (blurred and CLAHE-enhanced) grayscale image.
 * @param[out] out_original_color_img A copy of the original loaded color image.
 * @return bool True if preprocessing is successful, false otherwise (e.g., image not found).
 */
bool preprocess_image(const std::string &image_path, const Config &cfg,
                      cv::Mat &out_enhanced_img, cv::Mat &out_original_color_img)
{
    // Load the image in color
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return false;
    }
    out_original_color_img = img.clone(); // Store a copy of the original color image

    // Convert to grayscale
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

    // 1. Noise Reduction (Gaussian Blur)
    cv::Mat blurred_img;
    cv::GaussianBlur(gray_img, blurred_img, cfg.gaussian_blur_kernel_size, 0);

    // 2. Contrast Enhancement (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8)); // Standard CLAHE parameters
    clahe->apply(blurred_img, out_enhanced_img);

    // Display intermediate preprocessing steps if enabled
    if (cfg.display_intermediate_steps)
    {
        std::vector<std::pair<cv::Mat, std::string>> steps;
        steps.push_back({out_original_color_img, "Original"});
        steps.push_back({gray_img, "Grayscale"}); // Show the grayscale version
        steps.push_back({blurred_img, "Blurred"});
        steps.push_back({out_enhanced_img, "CLAHE Enhanced"});

        cv::Mat grid = createImageGrid(steps, cfg, std::min(static_cast<int>(steps.size()), cfg.grid_num_cols));
        if (!grid.empty())
        {
            cv::imshow("Preprocessing Steps", grid);
            cv::waitKey(0); // Wait for user input before closing
            cv::destroyWindow("Preprocessing Steps");
        }
    }
    return true;
}

/**
 * @brief Generates markers for the Watershed algorithm from a preprocessed image.
 * @param processed_img The preprocessed grayscale image (typically CLAHE enhanced).
 * @param cfg The global Config struct.
 * @return cv::Mat A CV_32S integer matrix where:
 *                 - 0 represents unknown regions.
 *                 - Positive integers (1, 2, ...) represent distinct foreground/background markers.
 *                 This format is directly usable by cv::watershed.
 */
cv::Mat generate_markers(const cv::Mat &processed_img, const Config &cfg)
{
    // 1. Thresholding to get a binary image
    // Adaptive thresholding is often better for images with varying illumination.
    // THRESH_BINARY_INV is used assuming the object of interest is brighter than the background
    // after preprocessing. If objects are darker, THRESH_BINARY might be needed.
    cv::Mat thresh;
    cv::adaptiveThreshold(processed_img, thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                          cv::THRESH_BINARY_INV, cfg.adaptive_thresh_block_size, cfg.adaptive_thresh_c);

    // 2. Morphological Opening (remove noise)
    // Opening = erosion followed by dilation. Removes small white noise specks.
    cv::Mat kernel_open = cv::getStructuringElement(cv::MORPH_ELLIPSE, cfg.morph_open_kernel_size);
    cv::Mat opening;
    cv::morphologyEx(thresh, opening, cv::MORPH_OPEN, kernel_open, cv::Point(-1, -1), 2); // 2 iterations

    // 3. Sure Background Area
    // Dilating the 'opening' result expands the foreground regions.
    // The area outside these expanded regions is considered sure background.
    cv::Mat kernel_dilate_bg = cv::getStructuringElement(cv::MORPH_ELLIPSE, cfg.morph_dilate_kernel_size);
    cv::Mat sure_bg;
    cv::dilate(opening, sure_bg, kernel_dilate_bg, cv::Point(-1, -1), 3); // 3 iterations

    // 4. Sure Foreground Area
    // Distance transform calculates the distance from each binary image pixel to the nearest zero pixel.
    // Peaks in the distance transform correspond to centers of foreground objects.
    cv::Mat dist_transform; // Output is CV_32F
    cv::distanceTransform(opening, dist_transform, cv::DIST_L2, 5); // L2 (Euclidean) distance, 5x5 mask

    // Threshold the distance map to get peaks (sure foreground)
    double minVal_dt, maxVal_dt;
    cv::minMaxLoc(dist_transform, &minVal_dt, &maxVal_dt); // Find max distance value
    cv::Mat sure_fg_float;
    cv::threshold(dist_transform, sure_fg_float, cfg.dist_transform_threshold_ratio * maxVal_dt, 255, cv::THRESH_BINARY);
    
    cv::Mat sure_fg_8u; // Convert to 8-bit unsigned for connectedComponents and morphological ops
    sure_fg_float.convertTo(sure_fg_8u, CV_8U);

    // Optional: Erode sure_fg to make markers smaller and more centered, reducing oversegmentation.
    cv::Mat kernel_erode_fg = cv::getStructuringElement(cv::MORPH_ELLIPSE, cfg.sure_fg_erode_kernel_size);
    cv::erode(sure_fg_8u, sure_fg_8u, kernel_erode_fg, cv::Point(-1, -1), 1); // 1 iteration

    // 5. Unknown Region (region between sure background and sure foreground)
    cv::Mat unknown;
    cv::subtract(sure_bg, sure_fg_8u, unknown); // sure_bg is larger, so unknown is white where sure_bg is white and sure_fg is black.

    // 6. Create Markers for Watershed
    // Label connected components in sure_fg. Background (0-pixels in sure_fg_8u) gets label 0.
    cv::Mat markers_from_cc; // Output is CV_32S
    int num_labels = cv::connectedComponents(sure_fg_8u, markers_from_cc, 8, CV_32S); // 8-connectivity

    // Adjust marker labels for cv::watershed:
    // - cv::connectedComponents: background=0, objects=1,2,...
    // - cv::watershed: unknown=0, one region type (e.g. background)=1, other objects=2,3,...
    cv::Mat markers = markers_from_cc + 1; // Shift all labels: background 0->1, components 1->2, 2->3, etc.
    markers.setTo(0, unknown == 255);      // Mark the unknown region with 0 for watershed.

    // Display intermediate marker generation steps if enabled
    if (cfg.display_intermediate_steps)
    {
        std::vector<std::pair<cv::Mat, std::string>> steps;
        steps.push_back({thresh, "Thresholded"});
        steps.push_back({opening, "Opening"});
        steps.push_back({sure_bg, "Sure Background"});
        steps.push_back({dist_transform, "Distance Transform"}); // prepareImageForGrid will normalize this CV_32F
        steps.push_back({sure_fg_8u, "Sure Foreground"});
        steps.push_back({unknown, "Unknown Region"});
        steps.push_back({applyColormapToMarkers(markers), "Initial Markers"}); // Show colormapped markers

        cv::Mat grid = createImageGrid(steps, cfg, std::min(static_cast<int>(steps.size()), cfg.grid_num_cols));
        if (!grid.empty())
        {
            cv::imshow("Marker Generation Steps", grid);
            cv::waitKey(0);
            cv::destroyWindow("Marker Generation Steps");
        }
    }
    return markers; // CV_32S matrix ready for cv::watershed
}

/**
 * @brief Applies the Watershed algorithm and visualizes the segmentation results.
 * @param original_color_img The original input color image.
 * @param preprocessed_gray_img The preprocessed grayscale image (kept for signature, not directly used by cv::watershed here).
 * @param[in,out] markers A CV_32S integer matrix of markers. cv::watershed modifies this in-place,
 *                        marking boundaries with -1.
 * @param cfg The global Config struct.
 * @param base_filename The base filename of the input image, used for naming the output saved grid.
 */
void apply_watershed_and_visualize(const cv::Mat &original_color_img,
                                   const cv::Mat &preprocessed_gray_img, // Unused in this specific watershed call
                                   cv::Mat &markers, 
                                   const Config &cfg,
                                   const std::string &base_filename)
{
    // Watershed algorithm expects a 3-channel image for processing.
    // It internally uses gradients. Using the original color image can provide richer gradient information.
    cv::Mat img_for_watershed = original_color_img.clone();
    cv::watershed(img_for_watershed, markers); // markers is modified in-place. Boundaries are set to -1.

    // --- Visualization Preparation ---
    // 1. Image with boundaries marked
    cv::Mat segmented_boundaries_img = original_color_img.clone();
    segmented_boundaries_img.setTo(cv::Scalar(0, 0, 255), markers == -1); // Mark boundaries in Red (BGR)

    // 2. Create a colormap for different segments
    std::set<int> unique_labels_set;
    for (int r = 0; r < markers.rows; ++r) {
        for (int c = 0; c < markers.cols; ++c) {
            unique_labels_set.insert(markers.at<int>(r, c));
        }
    }
    std::vector<int> unique_labels_vec(unique_labels_set.begin(), unique_labels_set.end());


    std::map<int, cv::Vec3b> colors;
    std::mt19937 rng(std::random_device{}()); // Mersenne Twister random number generator
    std::uniform_int_distribution<int> dist(50, 255); // Distribution for bright random colors

    for (int label : unique_labels_vec)
    {
        if (label == 0) { colors[label] = cv::Vec3b(0, 0, 0); }             // Unknown region (often black)
        else if (label == -1) { colors[label] = cv::Vec3b(0, 0, 255); }     // Watershed boundaries (Red)
        else { colors[label] = cv::Vec3b(dist(rng), dist(rng), dist(rng));} // Random color for segments
    }

    // 3. Image with colored segments (no blending yet)
    cv::Mat colored_segments_pure = cv::Mat::zeros(original_color_img.size(), CV_8UC3);
    for (int r = 0; r < markers.rows; ++r)
    {
        for (int c = 0; c < markers.cols; ++c)
        {
            int label = markers.at<int>(r, c);
            // Color only actual segments, not background (0) or boundaries (-1) directly here for pure segments
            if (label > 0 && colors.count(label)) 
            {
                colored_segments_pure.at<cv::Vec3b>(r, c) = colors[label];
            }
        }
    }
    
    // 4. Blended image: colored segments overlaid on original image
    cv::Mat blended_img;
    cv::addWeighted(colored_segments_pure, 0.5, original_color_img, 0.5, 0.0, blended_img);


    // --- Final Grid Display and Saving ---
    std::vector<std::pair<cv::Mat, std::string>> final_steps;
    final_steps.push_back({original_color_img, "Original"});
    final_steps.push_back({applyColormapToMarkers(markers), "Watershed Output"}); // Shows boundaries as -1 (often different color)
    final_steps.push_back({segmented_boundaries_img, "Boundaries Marked"});
    final_steps.push_back({colored_segments_pure, "Colored Segments"});
    final_steps.push_back({blended_img, "Blended Overlay"});

    cv::Mat final_grid = createImageGrid(final_steps, cfg, std::min(static_cast<int>(final_steps.size()), cfg.grid_num_cols));
    if (!final_grid.empty())
    {
        std::string final_grid_window_title = "Final Segmentation Results: " + base_filename;
        cv::imshow(final_grid_window_title, final_grid);

        // Save the final grid image
        std::string save_filename = cfg.output_grid_filename_prefix + "_" + base_filename + ".png";
        if (cv::imwrite(save_filename, final_grid))
        {
            std::cout << "Final grid saved to: " << save_filename << std::endl;
        }
        else
        {
            std::cerr << "Error: Could not save final grid to " << save_filename << std::endl;
        }
        cv::waitKey(0); // Wait for user input before closing the final display
        cv::destroyWindow(final_grid_window_title);
    }
}

// --- Utility and Main Functions ---

/**
 * @brief Creates a synthetic MRI-like image for demonstration purposes.
 * @param save_path Optional path to save the generated dummy image. If empty, not saved.
 * @return cv::Mat The generated dummy MRI image.
 */
cv::Mat create_dummy_mri(const std::string &save_path = "mri_sample_generated.png")
{
    cv::Mat dummy_img = cv::Mat::zeros(256, 256, CV_8UC3);
    dummy_img.setTo(cv::Scalar(30, 30, 30)); // Background (BGR)

    // Draw some "anatomical" structures and a "tumor"
    cv::circle(dummy_img, cv::Point(128, 128), 70, cv::Scalar(100, 100, 100), -1); // Larger "organ"
    cv::circle(dummy_img, cv::Point(100, 100), 20, cv::Scalar(180, 180, 180), -1); // Smaller "tumor" 1
    cv::circle(dummy_img, cv::Point(160, 170), 15, cv::Scalar(160, 160, 160), -1); // Smaller "tumor" 2

    // Add some noise
    cv::Mat noise(256, 256, CV_8UC3);
    cv::randu(noise, cv::Scalar::all(0), cv::Scalar::all(15)); // Uniform noise in range [0, 14]
    cv::add(dummy_img, noise, dummy_img);

    if (!save_path.empty())
    {
        if(cv::imwrite(save_path, dummy_img)){
            std::cout << "Generated a sample image: " << save_path << std::endl;
        } else {
            std::cerr << "Error: Could not save dummy image to " << save_path << std::endl;
        }
    }
    return dummy_img;
}

/**
 * @brief Extracts the base filename (without path and extension) from a full file path.
 * @param path The full path to the file.
 * @return std::string The base filename.
 */
std::string getBaseFilename(const std::string &path)
{
    // Find the last directory separator ('/' or '\')
    size_t last_slash_idx = path.find_last_of("/\\");
    std::string filename = (last_slash_idx == std::string::npos) ? path : path.substr(last_slash_idx + 1);

    // Find the last dot (extension separator)
    size_t dot_idx = filename.find_last_of('.');
    return (dot_idx == std::string::npos) ? filename : filename.substr(0, dot_idx);
}

/**
 * @brief Main function to execute the MRI segmentation pipeline.
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments. Expects image path as the first argument.
 * @return int 0 on success, -1 on failure.
 */
int main(int argc, char **argv)
{
    Config cfg; // Initialize configuration with default values
    std::string image_path_arg;
    std::string base_filename;

    // Parse command-line arguments
    if (argc > 1)
    {
        image_path_arg = argv[1];
    }
    else // Fallback to a dummy image if no path is provided
    {
        std::cout << "Usage: " << argv[0] << " <path_to_mri_image.jpg_or_png_or_tif>" << std::endl;
        std::cout << "Falling back to a default DEMO image path." << std::endl;
        create_dummy_mri("mri_sample_generated_cpp.png"); // Create and save dummy image
        image_path_arg = "mri_sample_generated_cpp.png";
    }
    base_filename = getBaseFilename(image_path_arg); // Get base filename for output naming

    // --- Execute Segmentation Pipeline ---
    cv::Mat preprocessed_img, original_color_img;
    if (!preprocess_image(image_path_arg, cfg, preprocessed_img, original_color_img))
    {
        std::cerr << "Image preprocessing failed. Exiting." << std::endl;
        return -1;
    }

    cv::Mat markers = generate_markers(preprocessed_img, cfg); // markers is CV_32S
    if (markers.empty())
    {
        std::cerr << "Marker generation failed. Exiting." << std::endl;
        return -1;
    }

    // Apply watershed and visualize/save results
    // Note: cv::watershed modifies markers in-place. If original markers are needed later, clone before passing.
    apply_watershed_and_visualize(original_color_img, preprocessed_img, markers, cfg, base_filename);

    std::cout << "Processing complete." << std::endl;
    return 0;
}