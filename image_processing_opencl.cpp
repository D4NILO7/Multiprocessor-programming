#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include<stdio.h>
#include<stdlib.h>
#include<lodepng.c>
#include<CL/cl.h>
#include<time.h>

#define MAX_SOURCE_SIZE (0x100000)
#define RESIZE_RATE 4

void show_platform_information();
void load_image(unsigned char** image, char* filename, unsigned* height, unsigned* width);
void save_image(unsigned char* image, char* filename, unsigned height, unsigned width);
void check_ret_value(cl_int ret, const char* error_message);

// to profile load and save
clock_t start_profiling();
void end_profiling(clock_t start, const char* fun_name);

// to profile kernel execution times
void kernel_profiling_info(cl_event event, const char* fun_name);


int main() {
	
	show_platform_information();

	// measuring the execution time of the whole process
	clock_t start = start_profiling();

	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_mem memobj = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platform_id = NULL;
	cl_int ret;
	cl_uint ret_num_platforms;

	cl_event event;

	cl_mem mem_left_input_image = NULL;
	cl_mem mem_left_resized_image = NULL;
	cl_mem mem_left_grayscaled_image = NULL;
	cl_mem mem_left_filtered_image = NULL;
	cl_mem mem_left_bordered_image = NULL;
	cl_mem mem_left_disp_image = NULL;

	cl_mem mem_right_input_image = NULL;
	cl_mem mem_right_resized_image = NULL;
	cl_mem mem_right_grayscaled_image = NULL;
	cl_mem mem_right_filtered_image = NULL;
	cl_mem mem_right_bordered_image = NULL;
	cl_mem mem_right_disp_image = NULL;

	cl_mem mem_post_processed_image = NULL;

	size_t globalWorkSize[2];

	FILE* fp;
	char fileName[] = "image_processing.cl";
	char* source_str;
	size_t source_size;

	char input_image_file_left[] = "C:/University_of_Oulu/MPP/OpenClProject/OpenClProject/images/im0.png";
	char input_image_file_right[] = "C:/University_of_Oulu/MPP/OpenClProject/OpenClProject/images/im1.png";
	char output_image_file[] = "C:/University_of_Oulu/MPP/OpenClProject/OpenClProject/images/output_image.png";
	
	unsigned char* left_image = 0;
	unsigned char* right_image = 0;
	unsigned width, height;

	load_image(&left_image, input_image_file_left, &height, &width);
	load_image(&right_image, input_image_file_right, &height, &width);

	// measuring the execution time of OpenCl execution (withou load and save) 
	printf("\nStarting the execution of OpenCL kernels\n");
	clock_t start_after_load = start_profiling();

	// Size of the output image
	unsigned output_height = height / RESIZE_RATE;
	unsigned output_width = width / RESIZE_RATE;


	// Step 1: Loading the source code containing the kernel
	fopen_s(&fp, fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel. \n");
		exit(1);
	}

	source_str = (char*)calloc(MAX_SOURCE_SIZE, 1);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);

	fclose(fp);

	// Step 2: Getting platform information
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	check_ret_value(ret, "Error getting platform IDs");


	// Step 3: Getting device information
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	check_ret_value(ret, "Error getting device ID");

	// Step 4: Creating an OpenCL context
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	check_ret_value(ret, "Failed to create OpenCL context");


	// Step 5: Creating a command queue
	command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
	check_ret_value(ret, "Failed to create command queue");

	// Step 6: Creating the OpenCL program
	program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);
	check_ret_value(ret, "Failed to create OpenCL program");

	// Step 7: Building OpenCL program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("Error: Could not build program\n");
		if (ret == CL_BUILD_PROGRAM_FAILURE) {
			/* Determine the size of the log*/
			size_t log_size;
			clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

			/* Allocate memory for log */
			char* log = (char*)malloc(log_size);

			/* Get the log */
			clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

			printf("%s\n", log);
		}
	}


	/*
		RESIZING IMAGE
	*/
	

	// Creating buffers for input and resized images
	mem_left_input_image = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * height * width * 4, NULL, &ret);
	mem_right_input_image = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * height * width * 4, NULL, &ret);

	mem_left_resized_image = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * output_height * output_width * 4, NULL, &ret);
	mem_right_resized_image = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * output_height * output_width * 4, NULL, &ret);
	
	// Writing the buffers
	ret = clEnqueueWriteBuffer(command_queue, mem_left_input_image, CL_TRUE, 0, sizeof(unsigned char) * width * height * 4, left_image, 0, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to transfer data to device memory\n");
		exit(1);
	}

	ret = clEnqueueWriteBuffer(command_queue, mem_right_input_image, CL_TRUE, 0, sizeof(unsigned char) * width * height * 4, right_image, 0, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to transfer data to device memory\n");
		exit(1);
	}

	// Creating OpenCl kernel for resizing the images
	kernel = clCreateKernel(program, "resize_image", &ret);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to create kernel resize\n");
		exit(1);
	}

	// Setting OpenCL Kernel Parameters for left image
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_left_input_image);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mem_left_resized_image);
	ret = clSetKernelArg(kernel, 2, sizeof(int), (void*)&height);
	ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)&width);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel arguments\n");
		exit(1);
	}

	// Executing OpenCL Kernel
	globalWorkSize[0] = output_height, globalWorkSize[1] = output_width;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &event);
	if (ret != CL_SUCCESS) {
		printf("Error: Could not execute kernel\n");
		exit(1);
	}
	kernel_profiling_info(event, "Resizing image");

	// Setting OpenCL Kernel Parameters for right image
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_right_input_image);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mem_right_resized_image);
	ret = clSetKernelArg(kernel, 2, sizeof(int), (void*)&height);
	ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)&width);

	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel arguments\n");
		exit(1);
	}

	// Executing OpenCL Kernel
	globalWorkSize[0] = output_height, globalWorkSize[1] = output_width;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &event);
	if (ret != CL_SUCCESS) {
		printf("Error: Could not execute kernel\n");
		exit(1);
	}
	kernel_profiling_info(event, "Resizing image");
	

	/*
		GRAYSCALING IMAGE
	*/

	// Creating buffers for grayscaled images
	mem_left_grayscaled_image = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * output_height * output_width, NULL, &ret);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to create memory buffer 2\n");
		exit(1);
	}
	mem_right_grayscaled_image = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * output_height * output_width, NULL, &ret);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to create memory buffer 2\n");
		exit(1);
	}

	// Creating OpenCl kernel for grayscaling the images
	kernel = clCreateKernel(program, "grayscale_image", &ret);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to create kernel\n");
		exit(1);
	}

	// Setting OpenCL Kernel Parameters for left image
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_left_resized_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 1\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mem_left_grayscaled_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument2 \n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&output_height);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 3\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&output_width);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 4\n");
		exit(1);
	}

	// Executing OpenCL Kernel
	globalWorkSize[0] = output_height, globalWorkSize[1] = output_width;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &event);
	if (ret != CL_SUCCESS) {
		printf("Error: Could not execute kernel\n");
		exit(1);
	}
	kernel_profiling_info(event, "Grayscaling image");


	// Setting OpenCL Kernel Parameters for right image
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_right_resized_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 1\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mem_right_grayscaled_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument2 \n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&output_height);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 3\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&output_width);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 4\n");
		exit(1);
	}

	// Executing OpenCL Kernel
	globalWorkSize[0] = output_height, globalWorkSize[1] = output_width;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &event);
	if (ret != CL_SUCCESS) {
		printf("Error: Could not execute kernel\n");
		exit(1);
	}
	kernel_profiling_info(event, "Grayscaling image");



	/*
		FILTERING IMAGE
	*/

	// Creating buffers for filtered images
	mem_left_filtered_image = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * output_height * output_width, NULL, &ret);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to create memory buffer 2\n");
		exit(1);
	}

	mem_right_filtered_image = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * output_height * output_width, NULL, &ret);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to create memory buffer 2\n");
		exit(1);
	}

	// Creating OpenCl kernel for filtering the images
	kernel = clCreateKernel(program, "filter_image", &ret);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to create kernel\n");
		exit(1);
	}

	// Setting OpenCL Kernel Parameters for left image
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_left_grayscaled_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 1\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mem_left_filtered_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument2 \n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&output_height);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 3\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&output_width);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 4\n");
		exit(1);
	}

	// Executing OpenCL Kernel
	globalWorkSize[0] = output_height, globalWorkSize[1] = output_width;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &event);
	if (ret != CL_SUCCESS) {
		printf("Error: Could not execute kernel\n");
		exit(1);
	}
	kernel_profiling_info(event, "Filtering image");


	// Setting OpenCL Kernel Parameters for right image
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_right_grayscaled_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 1\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mem_right_filtered_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument2 \n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&output_height);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 3\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&output_width);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 4\n");
		exit(1);
	}

	// Executing OpenCL Kernel
	globalWorkSize[0] = output_height, globalWorkSize[1] = output_width;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &event);
	if (ret != CL_SUCCESS) {
		printf("Error: Could not execute kernel\n");
		exit(1);
	}
	kernel_profiling_info(event, "Filtering image");



	/*
		ADDING BORDER TO THE IMAGE
	*/


	// 260 -> from the calib.txt
	unsigned max_disp = 260 / RESIZE_RATE;
	// size of the image after adding the border
	unsigned bordered_height = output_height + 2 * max_disp;
	unsigned bordered_width = output_width + 2 * max_disp;

	// Creating buffers for bordered images
	mem_left_bordered_image = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * bordered_height * bordered_width, NULL, &ret);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to create memory buffer\n");
		exit(1);
	}

	mem_right_bordered_image = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * bordered_height * bordered_width, NULL, &ret);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to create memory buffer\n");
		exit(1);
	}

	// Creating OpenCl kernel for adding the border
	kernel = clCreateKernel(program, "add_border", &ret);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to create kernel\n");
		exit(1);
	}

	// Setting OpenCL Kernel Parameters for left image
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_left_filtered_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 0\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mem_left_bordered_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 1\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&output_height);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 2\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&output_width);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 3\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&bordered_height);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 4\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&bordered_width);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 5\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 6, sizeof(cl_int), (void*)&max_disp);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 6\n");
		exit(1);
	}

	// Executing OpenCL Kernel
	globalWorkSize[0] = bordered_height, globalWorkSize[1] = bordered_width;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &event);
	if (ret != CL_SUCCESS) {
		printf("Error: Could not execute kernel\n");
		exit(1);
	}
	kernel_profiling_info(event, "Adding border to image");


	// Setting OpenCL Kernel Parameters for right image
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_right_filtered_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 0\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mem_right_bordered_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 1\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&output_height);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 2\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&output_width);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 3\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&bordered_height);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 4\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&bordered_width);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 5\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 6, sizeof(cl_int), (void*)&max_disp);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 6\n");
		exit(1);
	}

	// Executing OpenCL Kernel
	globalWorkSize[0] = bordered_height, globalWorkSize[1] = bordered_width;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &event);
	if (ret != CL_SUCCESS) {
		printf("Error: Could not execute kernel\n");
		exit(1);
	}
	kernel_profiling_info(event, "Adding border to image");


	/*
		CALCULATING ZNCC
	*/

	// Window size for zncc calculation
	const unsigned WIN_SIZE = 9;
	// left image first - like bool (1 == true, 0 == false)
	unsigned left_first = 1;

	// Creating buffers for disparity images
	mem_left_disp_image = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * output_height * output_width, NULL, &ret);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to create memory buffer 2\n");
		exit(1);
	}

	mem_right_disp_image = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * output_height * output_width, NULL, &ret);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to create memory buffer 2\n");
		exit(1);
	}

	// Creating OpenCl kernel for zncc calculation
	kernel = clCreateKernel(program, "calc_zncc", &ret);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to create kernel\n");
		exit(1);
	}

	// Setting OpenCL Kernel Parameters for left image
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_left_bordered_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 0\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mem_right_bordered_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 1 \n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&mem_left_disp_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 2\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&max_disp);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 3\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&WIN_SIZE);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 4\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&bordered_height);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 5\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 6, sizeof(cl_int), (void*)&bordered_width);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 6\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 7, sizeof(cl_int), (void*)&left_first);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 7\n");
		exit(1);
	}

	// Executing OpenCL Kernel
	globalWorkSize[0] = bordered_height, globalWorkSize[1] = bordered_width;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &event);
	if (ret != CL_SUCCESS) {
		printf("Error: Could not execute kernel\n");
		exit(1);
	}
	kernel_profiling_info(event, "ZNCC calculation");


	// Setting OpenCL Kernel Parameters for right image
	left_first = 0;

	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_right_bordered_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 0\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mem_left_bordered_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 1 \n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&mem_right_disp_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 2\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&max_disp);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 3\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&WIN_SIZE);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 4\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&bordered_height);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 5\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 6, sizeof(cl_int), (void*)&bordered_width);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 6\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 7, sizeof(cl_int), (void*)&left_first);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 7\n");
		exit(1);
	}

	// Executing OpenCL Kernel
	globalWorkSize[0] = bordered_height, globalWorkSize[1] = bordered_width;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &event);
	if (ret != CL_SUCCESS) {
		printf("Error: Could not execute kernel\n");
		exit(1);
	}
	kernel_profiling_info(event, "ZNCC calculation");



	/*
		CROSS-CHECKING IMAGE
	*/

	// final image memory allocation
	unsigned char* post_processed_image = (unsigned char*)malloc(sizeof(unsigned char) * (output_height * output_width));
	// treshold value for cross checking
	unsigned treshold = 8;

	// Creating buffer for the final post processed image
	mem_post_processed_image = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * output_height * output_width, NULL, &ret);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to create memory buffer 2\n");
		exit(1);
	}

	// Creating OpenCl kernel for cross checking the images
	kernel = clCreateKernel(program, "cross_check_image", &ret);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to create kernel\n");
		exit(1);
	}

	// Setting OpenCL Kernel Parameters for cross checking
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_left_disp_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 0\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mem_right_disp_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 1 \n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&mem_post_processed_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 2\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&output_height);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 3\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&output_width);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 4\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&treshold);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 5\n");
		exit(1);
	}

	// Executing OpenCl kernel
	globalWorkSize[0] = output_height, globalWorkSize[1] = output_width;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &event);
	if (ret != CL_SUCCESS) {
		printf("Error: Could not execute kernel\n");
		exit(1);
	}
	kernel_profiling_info(event, "Cross checking image");



	/*
		OCCLUSION FILLING IMAGE
	*/

	// max window size for occlusion filling
	unsigned max_window_size = 5;

	// Creating OpenCl kernel for occlusion filling
	kernel = clCreateKernel(program, "occlusion_fill_nearest", &ret);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to create kernel\n");
		exit(1);
	}

	// Setting OpenCL Kernel Parameters for occlusion filling
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_post_processed_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 0\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&output_height);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 1\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&output_width);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 2\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&max_window_size);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 3\n");
		exit(1);
	}

	// Executing the kernel
	globalWorkSize[0] = output_height, globalWorkSize[1] = output_width;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &event);
	if (ret != CL_SUCCESS) {
		printf("Error: Could not execute kernel\n");
		exit(1);
	}
	kernel_profiling_info(event, "Occlusion filling image");


	/*
		NORMALIZING IMAGE
	*/
	
	// Creating OpenCl kernel for normalizing the image
	kernel = clCreateKernel(program, "normalize_image", &ret);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to create kernel\n");
		exit(1);
	}

	// Setting OpenCL Kernel Parameters for normalization
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_post_processed_image);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 0\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&output_height);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 1\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&output_width);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 2\n");
		exit(1);
	}
	ret = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&max_disp);
	if (ret != CL_SUCCESS) {
		printf("Error: Failed to set kernel argument 3\n");
		exit(1);
	}

	// Executing the kernel
	globalWorkSize[0] = output_height, globalWorkSize[1] = output_width;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &event);
	if (ret != CL_SUCCESS) {
		printf("Error: Could not execute kernel\n");
		exit(1);
	}
	kernel_profiling_info(event, "Normalizing");

	// Reading back the normalized image into post_processed_image
	ret = clEnqueueReadBuffer(command_queue, mem_post_processed_image, CL_TRUE, 0, sizeof(unsigned char) * output_height * output_width, post_processed_image, 0, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("Error: Could not copy buffer %d\n", ret);
		exit(1);
	}

	// finishing the profiling without loading and saving
	printf("Finishing the execution of OpenCL kernels\n");
	end_profiling(start_after_load, "Opencl process without load and save");
	printf("\n");
	
	
	/*
		SAVING IMAGE
	*/
	save_image(post_processed_image, output_image_file, output_height, output_width);



	/*
		CLEANUP
	*/

	ret = clReleaseMemObject(mem_left_input_image);
	ret = clReleaseMemObject(mem_right_input_image);
	ret = clReleaseMemObject(mem_left_resized_image);
	ret = clReleaseMemObject(mem_right_resized_image);
	ret = clReleaseMemObject(mem_left_grayscaled_image);
	ret = clReleaseMemObject(mem_right_grayscaled_image);
	ret = clReleaseMemObject(mem_left_filtered_image);
	ret = clReleaseMemObject(mem_right_filtered_image);
	ret = clReleaseMemObject(mem_left_bordered_image);
	ret = clReleaseMemObject(mem_right_bordered_image);
	ret = clReleaseMemObject(mem_left_disp_image);
	ret = clReleaseMemObject(mem_right_disp_image);
	ret - clReleaseMemObject(mem_post_processed_image);

	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	free(source_str);
	free(left_image);
	free(right_image);
	free(post_processed_image);

	// finishing the profiling
	end_profiling(start, "the whole process");
	//printf("all good\n");
	return 0;
}

void kernel_profiling_info(cl_event event, const char* fun_name) {
	cl_ulong start_time, end_time;
	clWaitForEvents(1, &event);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
	double execution_time = (double)(end_time - start_time) / 1000000000.0;
	printf(">>> OpenCL execution time of %s: %f s\n", fun_name, execution_time);
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

clock_t start_profiling() {
	clock_t start = clock();
	return start;
}

void end_profiling(clock_t start, const char* fun_name) {
	clock_t end = clock();
	double gpu_time = ((double)(end - start)) / 1000;
	printf("Execution time of %s: %f s\n", fun_name, gpu_time);
}

void check_ret_value(cl_int ret, const char* error_message) {
	if (ret != CL_SUCCESS) {
		printf("Error: %s %d\n", error_message, ret);
		exit(1);
	}
}

void show_platform_information() {

	cl_int err;					//storing error code	
	cl_device_id device;
	cl_platform_id platform;
	cl_uint num_platforms;

	//retrieves the list of available platforms
	err = clGetPlatformIDs(1, &platform, &num_platforms);
	if (err != CL_SUCCESS) {
		printf("Error getting platform IDs\n");
		exit(1);
	}
	printf("Number of platforms: %d\n", num_platforms);

	// CL_PLATFORM_VENDOR -> retrieves the vendor who created the platform (e.g. Intel, NVIDIA)
	char vendor[128];
	err = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
	if (err != CL_SUCCESS) {
		printf("Error getting platform vendor\n");
		exit(1);
	}
	printf("Platform Vendor: %s\n", vendor);

	// CL_PLATFORM_NAME -> retrieves the name of the OpenCL platform
	char name[128];
	err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name), name, NULL);
	if (err != CL_SUCCESS) {
		printf("Error getting platform name\n");
		exit(1);
	}
	printf("Platform Name: %s\n", name);

	// CL_PLATFORM_VERSION -> retrieves the version of the OpenCL platform
	char version[128];
	err = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(version), version, NULL);
	if (err != CL_SUCCESS) {
		printf("Error getting platform version\n");
		exit(1);
	}
	printf("Platform Version: %s\n", version);

	// retrieves the list of OpenCL devices available on the platform
	// Tried to do it with CL_DEVICE_TYPE_CPU or /ALL but there was an error
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if (err != CL_SUCCESS) {
		printf("Error getting device ID\n");
		exit(1);
	}

	// CL_DEVICE_NAME -> retrieves the name of the device
	char device_name[128];
	err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
	if (err != CL_SUCCESS) {
		printf("Error getting device name\n");
		exit(1);
	}
	printf("Device Name: %s\n", device_name);

	// CL_DEVICE_LOCAL_MEM_TYPE -> retrives the type of local memory supported by the device (either Local or Global)
	cl_device_local_mem_type local_mem_type;
	err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
	if (local_mem_type == 0) {
		printf("Local memory type: CL_LOCAL\n");
	}
	else if (local_mem_type == 1) {
		printf("Local memory type: CL_GLOBAL\n");
	}
	else if (err != CL_SUCCESS) {
		printf("ERROR getting the local memory type\n");
	}

	// CL_DEVICE_LOCAL_MEM_SIZE -> retrives the size of the local memory available per compute unit (in bytes)
	cl_ulong local_mem_size;
	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
	printf("Local memory size: %lu bytes\n", local_mem_size);

	// CL_DEVICE_MAX_COMPUTE_UNITS -> retrives the number of parallel compute units available
	cl_uint num_compute_units;
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(num_compute_units), &num_compute_units, NULL);
	if (err != CL_SUCCESS) {
		printf("Error getting device info\n");
		exit(1);
	}
	printf("Number of parallel compute units: %u\n", num_compute_units);

	// CL_DEVICE_MAX_CLOCK_FREQUENCY -> retrives an integer value representing the maximum clock frequency
	cl_uint max_clock_frequency;
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_clock_frequency), &max_clock_frequency, NULL);
	if (err != CL_SUCCESS) {
		printf("Error getting device info\n");
		exit(1);
	}
	printf("Maximum clock frequency: %u\n", max_clock_frequency);

	// CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE -> retrives an integer value representing the maximum size of a constant buffer
	cl_ulong max_constant_buffer_size;
	clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(max_constant_buffer_size), &max_constant_buffer_size, NULL);
	printf("Maximum constant buffer size: %lu bytes\n", max_constant_buffer_size);

	//CL_DEVICE_MAX_WORK_GROUP_SIZE -> retrives an integer value representing the maximum number of work-items that can be included in a single work-group
	size_t max_work_group_size;
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
	printf("Maximum work-group size: %zu\n", max_work_group_size);

	// CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS -> retrives an integer value representing the maximum number of dimensions that can be specified for a work-group
	cl_uint max_work_item_dimensions;
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dimensions), &max_work_item_dimensions, NULL);
	if (err != CL_SUCCESS) {
		printf("Error getting device info\n");
		exit(1);
	}
	printf("Maximum work item dimensions: %u\n", max_work_item_dimensions);

	// CL_DEVICE_MAX_WORK_ITEM_SIZES -> retrives an array of three values, which represent the maximum size of each dimension of a work-group (x, y, z dimension)
	size_t max_work_item_sizes[3];
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), &max_work_item_sizes, NULL);
	printf("Maximum work-item sizes: (%zu, %zu, %zu)\n", max_work_item_sizes[0], max_work_item_sizes[1], max_work_item_sizes[2]);

	printf("\n");
}

