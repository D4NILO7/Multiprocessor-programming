#include<stdio.h>
#include<stdlib.h>
#include<lodepng.c>
#include<time.h>

#define SCALE_RATE 4

void load_image(unsigned char** image, char* file_name, unsigned* height, unsigned* width);
unsigned char* resize_image(unsigned char* image, unsigned height, unsigned width);
float* grayscale_image(unsigned char* image, unsigned height, unsigned width);
unsigned char* filter_image(float* image, unsigned height, unsigned width);
void save_image(unsigned char* image, char* filename, unsigned height, unsigned width);
unsigned char* calc_zncc(unsigned char* image_left, unsigned char* image_right, unsigned max_disp, unsigned height, unsigned width, bool left_first);
unsigned char* add_border(unsigned char* image, int max_disp, unsigned height, unsigned width);
void normalize_image(unsigned char* image, unsigned height, unsigned width, unsigned max_disp);
unsigned char* cross_check_image(unsigned char* left_disp_image, unsigned char* right_disp_image, unsigned height, unsigned width, unsigned threshold);
void occlusion_fill_nearest(unsigned char* disp_image, unsigned height, unsigned width);

clock_t start_profiling();
void end_profiling(clock_t start, const char* fun_name);

int main() {

	// measuring the execution time of the whole process
	clock_t start = start_profiling();

	unsigned error;
	unsigned char* left_image = 0;
	unsigned char* right_image = 0;

	char left_image_filename[] = "C:/University_of_Oulu/MPP/OpenClProject/OpenClProject/images/im0.png";
	char right_image_filename[] = "C:/University_of_Oulu/MPP/OpenClProject/OpenClProject/images/im1.png";
	char output_filename[] = "C:/University_of_Oulu/MPP/OpenClProject/OpenClProject/images/output.png";


	unsigned width, height;
	// loading the images
	load_image(&left_image, left_image_filename, &height, &width);
	load_image(&right_image, right_image_filename, &height, &width);

	//printf("Width: %d, Height: %d\n", width, height);

	unsigned output_width = width / SCALE_RATE, output_height = height / SCALE_RATE;

	//printf("After division\n");

	// resizing the images
	unsigned char* left_resized_image = resize_image(left_image, height, width);
	unsigned char* right_resized_image = resize_image(right_image, height, width);

	//printf("After resizing\n");

	// grayscaling the images
	float* left_grayscaled_image = grayscale_image(left_resized_image, output_height, output_width);
	float* right_grayscaled_image = grayscale_image(right_resized_image, output_height, output_width);

	//printf("After grayscaling\n");
	
	// filtering the images using gaussian blur
	unsigned char* left_filtered_image = filter_image(left_grayscaled_image, output_height, output_width);
	unsigned char* right_filtered_image = filter_image(right_grayscaled_image, output_height, output_width);

	char filtered_filename[] = "C:/University_of_Oulu/MPP/OpenClProject/OpenClProject/images/filtered_image.png";
	//save_image(left_filtered_image, filtered_filename, output_height, output_width);

	// 260 -> from the calib.txt
	int max_disp = 260 / SCALE_RATE;
	// size of the image after adding the border
	int bordered_height = output_height + 2 * max_disp;
	int bordered_width = output_width + 2 * max_disp;

	// adding border to the images so zncc algorithm can work properly
	unsigned char* left_bordered_image = add_border(left_filtered_image, max_disp, output_height, output_width);
	unsigned char* right_bordered_image = add_border(right_filtered_image, max_disp, output_height, output_width);
	
	// creating the disparity maps of both images
	unsigned char* left_disp_image = calc_zncc(left_bordered_image, right_bordered_image, max_disp, bordered_height, bordered_width, true);
	unsigned char* right_disp_image = calc_zncc(right_bordered_image, left_bordered_image, max_disp, bordered_height, bordered_width, false);

	char left_disp_filename[] = "C:/University_of_Oulu/MPP/OpenClProject/OpenClProject/images/left_disp_image.png";
	//normalize_image(left_disp_image, output_height, output_width, max_disp);
	//save_image(left_disp_image, left_disp_filename, output_height, output_width);

	char right_disp_filename[] = "C:/University_of_Oulu/MPP/OpenClProject/OpenClProject/images/right_disp_image.png";
	//save_image(right_disp_image, right_disp_filename, output_height, output_width);

	unsigned char* cross_checked_image = cross_check_image(left_disp_image, right_disp_image, output_height, output_width, 8);
	occlusion_fill_nearest(cross_checked_image, output_height, output_width);

	normalize_image(cross_checked_image, output_height, output_width, max_disp);
		
	char cross_checked_filename[] = "C:/University_of_Oulu/MPP/OpenClProject/OpenClProject/images/cross_checked_image.png";

	save_image(cross_checked_image, cross_checked_filename, output_height, output_width);


	//printf("After saving\n");

	free(left_image);
	free(right_image);
	free(left_resized_image);
	free(right_resized_image);
	free(left_grayscaled_image);
	free(right_grayscaled_image);
	free(left_filtered_image);
	free(right_filtered_image);

	free(left_bordered_image);
	free(right_bordered_image);
	 
	free(left_disp_image);
	free(right_disp_image);

	free(cross_checked_image);

	// finishing the profiling
	end_profiling(start, "the whole process");

	return 0;
}

unsigned char* calc_zncc(unsigned char* image_left, unsigned char* image_right, unsigned max_disp, unsigned height, unsigned width, bool left_first) {
	clock_t start = start_profiling();

	const int WIN_SIZE = 9;
	float max_zncc = 0.0;
	int best_disp = 0;

	unsigned char* disp_image = (unsigned char*)malloc(sizeof(unsigned char) * ((height - max_disp) * (width - max_disp)));

	for (int j = max_disp ; j < height - max_disp; j++) {
		for (int i = max_disp; i < width - max_disp; i++) {
			max_zncc = 0.0;
			best_disp = 0;
			for (int d = 0; d < max_disp; d++) {
				float sum = 0.0;
				for (int wy = 0; wy < WIN_SIZE; wy++) {
					for (int wx = 0; wx < WIN_SIZE; wx++) {
						int y = j + wy;
						int x = i + wx;
						float val = (float) image_left[y * width + x];
						sum += val;
					}
				}
				float mean = sum / (WIN_SIZE * WIN_SIZE * 2);

				float num = 0.0f;
				float zncc_left = 0.0f;
				float zncc_right = 0.0f;
				for (int wy = 0; wy < WIN_SIZE; wy++) {
					for (int wx = 0; wx < WIN_SIZE; wx++) {
						int x = i + wx;
						int y = j + wy;
						float left_val = image_left[y * width + x] - mean;
						float right_val;
						if (left_first) {
							right_val = image_right[y * width + x - d] - mean;
						}
						else {
							right_val = image_right[y * width + x + d] - mean;
						}

						num += left_val * right_val;
						zncc_left += left_val * left_val;
						zncc_right += right_val * right_val;
					}
				}
				float zncc = num / sqrt(zncc_left * zncc_right);
				if (zncc > max_zncc) {
					max_zncc = zncc;
					best_disp = d;
				}
			}
			disp_image[(j - max_disp) * (width - (2 * max_disp)) + (i - max_disp)] = (unsigned char)best_disp;
		}
	}
	end_profiling(start, "Calculating ZNCC");
	return disp_image;
}


void normalize_image(unsigned char* image,unsigned height, unsigned width, unsigned max_disp) {
	
	clock_t start = start_profiling();
	// Normalize the pixel values
	for (unsigned row = 0; row < height; row++) {
		for (unsigned col = 0; col < width; col++) {
			unsigned idx = row * width + col;
			image[idx] = (image[idx] * 255) / max_disp;
		}
	}

	end_profiling(start, "Normalizig");
}

unsigned char* cross_check_image(unsigned char* left_disp_image, unsigned char* right_disp_image, unsigned height, unsigned width, unsigned threshold) {
	
	clock_t start = start_profiling();

	unsigned char* cross_image = (unsigned char*)malloc(sizeof(unsigned char) * (height * width));
	for (unsigned row = 0; row < height; row++) {
		for (unsigned col = 0; col < width; col++) {
			unsigned idx = row * width + col;
			unsigned left_pix = left_disp_image[idx];
			unsigned right_pix = right_disp_image[idx - left_pix];
			if (abs((int)(left_pix - right_pix)) > threshold) {
				cross_image[idx] = 0;
			} else {
				cross_image[idx] = left_pix;
			}
		}
	}
	end_profiling(start, "Cross-checking");
	return cross_image;
}

void occlusion_fill_nearest(unsigned char* disp_image, unsigned height, unsigned width) {
	
	clock_t start = start_profiling();
	
	for (unsigned row = 0; row < height; row++) {
		for (unsigned col = 0; col < width; col++) {
			unsigned idx = row * width + col;
			if (disp_image[idx] == 0) {
				int win_size = 1;
				while (win_size < 5) {
					for (int fil_row = -win_size; fil_row <= win_size; fil_row++) {
						for (int fil_col = -win_size; fil_col <= win_size; fil_col++) {
							if (fil_row == 0 && fil_col == 0) {
								continue;
							}
							int pos_row = row + fil_row;
							int pos_col = col + fil_col;
							if (pos_row >= 0 && pos_row < height && pos_col >= 0 && pos_col < width) {
								unsigned fil_idx = pos_row * width + pos_col;
								if (disp_image[fil_idx] != 0) {
									disp_image[idx] = disp_image[fil_idx];
									break;
								}
							}
						}
						if (disp_image[idx] != 0) {
							break;
						}
					}
					if (disp_image[idx] != 0) {
						break;
					}
					win_size++;
				}
			}
		}
	}
	end_profiling(start, "Occlusion filling");
}

void load_image(unsigned char** image, char* filename, unsigned* height, unsigned* width) {

	clock_t start = start_profiling();

	unsigned error = lodepng_decode32_file(image, width, height, filename);
	if (error) {
		printf("error %u: %s\n", error, lodepng_error_text(error));
		exit(1);
	}
	end_profiling(start, "Loading image");
}

unsigned char* resize_image(unsigned char* image, unsigned height, unsigned width) {
	clock_t start = start_profiling();

	unsigned output_height = height / 4, output_width = width / 4;
	unsigned char* resized_image = (unsigned char*)malloc(output_width * output_height * 4);
	if (!resized_image) {
		printf("Error: failed to allocate memory for output image\n");
		exit(1);
	}


	/* Image resizing */
	for (unsigned row = 0; row < output_height; row++) {
		for (unsigned col = 0; col < output_width; col++) {
			unsigned input_idx = (row * 4) * width * 4 + (col * 4) * 4;
			unsigned output_idx = row * output_width * 4 + col * 4;
			resized_image[output_idx] = image[input_idx];
			resized_image[output_idx + 1] = image[input_idx + 1];
			resized_image[output_idx + 2] = image[input_idx + 2];
			resized_image[output_idx + 3] = image[input_idx + 3];
		}
	}
	end_profiling(start, "Resizing image");
	return resized_image;
}

float* grayscale_image(unsigned char* image, unsigned height, unsigned width) {
	clock_t start = start_profiling();

	float* grayscaled_image = (float*)malloc(sizeof(float) * height * width);
	for (unsigned row = 0; row < height; row++) {
		for (unsigned col = 0; col < width; col++) {
			unsigned idx = row * 4 * width + col * 4;
			unsigned char r = image[idx];
			unsigned char g = image[idx + 1];
			unsigned char b = image[idx + 2];
			unsigned char a = image[idx + 3];

			/* convert to grayscale and store in image */
			float gray = 0.2126 * r + 0.7152 * g + 0.0722 * b;
			grayscaled_image[row * width + col] = gray;

		}
	}
	end_profiling(start, "Grayscaling image");

	return grayscaled_image;
}

unsigned char* filter_image(float* image, unsigned height, unsigned width) {
	clock_t start = start_profiling();
	unsigned char* filtered_image = (unsigned char*)malloc(sizeof(unsigned char) * (height * width));
	if (!filtered_image) {
		printf("Error: failed to allocate memory for gaussian blur image\n");
		exit(1);
	}
	unsigned filter[5][5] = { {1, 1, 1, 1, 1},
							  {1, 1, 1, 1, 1},
							  {1, 1, 1, 1, 1},
							  {1, 1, 1, 1, 1},
							  {1, 1, 1, 1, 1}
	};

	// rows
	for (int row = 2; row < height - 2; row++) {
		// columns
		for (int col = 2; col < width - 2; col++) {
			float sum = 0.0;
			unsigned idx = row * width + col;
			// rows and cols in filter
			for (int fil_row = 0; fil_row < 5; fil_row++) {
				for (int fil_col = 0; fil_col < 5; fil_col++) {
					unsigned pos_row = row + fil_row - 2;
					unsigned pos_col = col + fil_col - 2;

					unsigned pixel_idx = pos_row * width + pos_col;
					sum += image[pixel_idx] * filter[fil_row][fil_col];
				}
			}

			float blurred_pixel = sum / 25;
			filtered_image[idx] = (unsigned char) blurred_pixel;
		}
	}

	end_profiling(start, "Filtering image");

	return filtered_image;
}



void save_image(unsigned char* image, char* filename, unsigned height, unsigned width) {
	clock_t start = start_profiling();
	unsigned error = lodepng_encode_file(filename, image, width, height, LCT_GREY, 8);
	if (error) {
		printf("ERROR\n");
		printf("error %u: %s\n", error, lodepng_error_text(error));
		exit(1);
	}
	end_profiling(start, "Saving image");
}

unsigned char* add_border(unsigned char* image, int max_disp, unsigned height, unsigned width) {

	clock_t start = start_profiling();
	unsigned error;

	// Define border size
	int border_size = max_disp;

	// Calculate the size of the padded image (multiplying with two becouse there is a border on both sides)
	unsigned padded_width = width + 2 * border_size;
	unsigned padded_height = height + 2 * border_size;

	// Allocate memory for the padded image
	unsigned char* padded_image = (unsigned char*)malloc(padded_width * padded_height * sizeof(unsigned char));

	// Initialize the padded image to black
	memset(padded_image, 0, padded_width * padded_height * sizeof(unsigned char));

	// Copy the original image to the center of the padded image
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			padded_image[(y + border_size) * padded_width + (x + border_size)] = image[y * width + x];
		}
	}

	end_profiling(start, "Adding border");
	return padded_image;
}


clock_t start_profiling() {
	clock_t start = clock();
	return start;
}

void end_profiling(clock_t start, const char* fun_name) {
	clock_t end = clock();
	double cpu_time = ((double)(end - start)) / 1000;
	printf("C execution time of %s: %f s\n", fun_name, cpu_time);
} 
