__kernel void resize_image(__global const unsigned char* image,
    __global unsigned char* resized_image,
    const unsigned int height,
    const unsigned int width
    ) {

    const unsigned int out_width = width / 4;
    const unsigned int out_height = height / 4;

    const unsigned int row = get_global_id(0);
    const unsigned int col = get_global_id(1);

        const unsigned int input_idx = (row * 4) * width *4 + (col * 4)*4;
        const unsigned int output_idx = row * out_width *4 + col*4;
        resized_image[output_idx] = image[input_idx];
        resized_image[output_idx + 1] = image[input_idx + 1];
        resized_image[output_idx + 2] = image[input_idx + 2];
        resized_image[output_idx + 3] = image[input_idx + 3];
}

__kernel void grayscale_image(__global unsigned char* input_image, __global float* output_image, const unsigned int height, const unsigned int width) {
    const unsigned int row = get_global_id(0);
    const unsigned int col = get_global_id(1);

        const unsigned int idx = row * 4 * width + col * 4;
        unsigned char r = input_image[idx];
        unsigned char g = input_image[idx + 1];
        unsigned char b = input_image[idx + 2];
    
        float gray = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        output_image[row * width + col] = gray;
}

__kernel void filter_image(__global float* input_image,__global unsigned char* output_image, const unsigned int height, const unsigned int width){
    const unsigned int row = get_global_id(0);
    const unsigned int col = get_global_id(1);

    const float filter[5][5] = { {1, 1, 1, 1, 1},
							        {1, 1, 1, 1, 1},
							        {1, 1, 1, 1, 1},
							        {1, 1, 1, 1, 1},
							        {1, 1, 1, 1, 1}
	};

    if((row < 2 || row >= height-2) || (col < 2 || col >= width-2)){
        return;
    }

    float sum = 0.0;
    unsigned idx = row * width + col;
    for (int fil_row = 0; fil_row < 5; fil_row++) {
		for (int fil_col = 0; fil_col < 5; fil_col++) {
			unsigned pos_row = row + fil_row - 2;
			unsigned pos_col = col + fil_col - 2;

            unsigned pixel_idx = pos_row * width + pos_col;
			sum += input_image[pixel_idx] * filter[fil_row][fil_col];
        }
    }
    float blurred_pixel = sum / 25;
    output_image[idx] = (unsigned char) blurred_pixel;
}


__kernel void add_border(__global unsigned char* image, __global unsigned char* bordered_image, unsigned height, unsigned width, unsigned bordered_height, unsigned bordered_width, unsigned max_disp) {
    const unsigned int row = get_global_id(0);
    const unsigned int col = get_global_id(1);

    if (col < bordered_width && row < bordered_height) {
        unsigned idx = row * bordered_width + col;
        if (col >= max_disp && col < width + max_disp && row >= max_disp && row < height + max_disp) {
            bordered_image[idx] = image[(row - max_disp) * width + (col - max_disp)];
        }
        else {
            bordered_image[idx] = 0;
        }
    }
}

__kernel void calc_zncc(__global const unsigned char* image_left, __global const unsigned char* image_right,
                         __global unsigned char* disp_image, const unsigned max_disp, const unsigned win_size,
                         const unsigned height, const unsigned width, const unsigned left_first) {
   
    unsigned int row = get_global_id(0);
    unsigned int col = get_global_id(1);
    
    if (row >= max_disp && row < height - max_disp && col >= max_disp && col < width - max_disp) {
        float max_zncc = 0.0f;
        int best_disp = 0;

        for (int d = 0; d < max_disp; d++) {
            float sum = 0.0f;
            for (int wy = 0; wy < win_size; wy++) {
                for (int wx = 0; wx < win_size; wx++) {
                    int y = row + wy;
                    int x = col + wx;
                    float val = (float) image_left[y * width + x];
                    sum += val;
                }
            }
            float mean = sum / (win_size * win_size * 2);
  
            float num = 0.0;
            float zncc_left = 0.0;
            float zncc_right = 0.0;

            for (int wy = 0; wy < win_size; wy++) {
                for (int wx = 0; wx < win_size; wx++) {
                    int x = col + wx;
                    int y = row + wy;
                    float left_val = image_left[y * width + x] - mean;
                    float right_val;
                    if (left_first == 1) {
                        right_val = image_right[y * width + x - d] - mean;
                    } else {
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
        disp_image[(row - max_disp) * (width - (2 * max_disp)) + (col - max_disp)] = (unsigned char)best_disp;
    }
}

__kernel void cross_check_image(__global const unsigned char* image_left, __global const unsigned char* image_right,
                                __global unsigned char* cross_checked_image, const unsigned height,
                                const unsigned width, const unsigned threshold) {
 
    unsigned int row = get_global_id(0);
    unsigned int col = get_global_id(1);

    unsigned idx = row * width + col;
	unsigned left_pix = image_left[idx];
	unsigned right_pix = image_right[idx - left_pix];
	if (abs((int)(left_pix - right_pix)) > threshold) {
		cross_checked_image[idx] = 0;
	} else {
		cross_checked_image[idx] = left_pix;
	}
}

__kernel void occlusion_fill_nearest(__global unsigned char* disp_image, const unsigned height, const unsigned width, const unsigned max_win_size){
    unsigned int row = get_global_id(0);
    unsigned int col = get_global_id(1);
    unsigned idx = row * width + col;
	if (disp_image[idx] == 0) {
		int win_size = 1;
		while (win_size < max_win_size) {
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

__kernel void normalize_image(__global unsigned char* image, const unsigned height, const unsigned width, const unsigned max_disp){
    unsigned int row = get_global_id(0);
    unsigned int col = get_global_id(1);
    
    unsigned idx = row * width + col;
	image[idx] = (image[idx] * 255) / max_disp;
}
