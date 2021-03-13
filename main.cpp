#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "cl2.hpp"
#include <iostream>
#include <vector>
#include <fstream>


#define ROWS 512
#define GENERAL_SIZE 512
#define COLUMNS 512

#define DEVICE_NUMBER 2

#define BLOCK_SIZE 16

using namespace std;

string read_kernel(string file_name) {
    ifstream input_file(file_name);
    if (!input_file.is_open()) {
        cout << "Error opening file:" << file_name <<endl;
        exit(EXIT_FAILURE);
    }
    return string((istreambuf_iterator<char>(input_file)), istreambuf_iterator<char>());
}

void get_random_matrix(float matrix[], int rows, int columns) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            matrix[i * columns + j] = rand() / (float)RAND_MAX;
        }
    }
}

void print_matrix(float matrix[], int rows, int columns) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            cout << matrix[i * columns + j] << " ";
        }
        cout << endl;
    }
}

void multiplication_matrix(float matrix_a[ROWS * GENERAL_SIZE], float matrix_b[GENERAL_SIZE * COLUMNS], float result[ROWS * COLUMNS]) {
    for(int i = 0; i < ROWS; i++) {
        for(int j = 0; j < COLUMNS; j++) {
            float res = 0.0;
            for(int k = 0; k < GENERAL_SIZE; k++) {
                res += matrix_a[i * GENERAL_SIZE + k] * matrix_b[k * COLUMNS + j];
            }
            result[i * COLUMNS + j] = res;
        }
    }
}

bool check_answ(float matrix_checking[], float matrix_verifiable[], int rows, int columns) {
    int counter = 0;
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < columns; j++) {
            if (matrix_checking[i * columns + j] == matrix_verifiable[i * columns + j]) {
                counter ++;
            }
        }
    }
    return counter == rows * columns ? true : false;
}

cl::Program get_program(cl::Device &device, cl::Context &context, string kernel_path) {
    string kernel_source = read_kernel(kernel_path);
    cl::Program program(context, kernel_source, true);
    if (program.build({ device }) != CL_SUCCESS) {
        cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
        exit(EXIT_FAILURE);
    }
    return program;
}


int main() {
    
    float a[ROWS * GENERAL_SIZE];
    float b[GENERAL_SIZE * COLUMNS];
    float res[ROWS * COLUMNS];
    float check[ROWS * COLUMNS];
    
    
    get_random_matrix(a, ROWS, GENERAL_SIZE);
    get_random_matrix(b, GENERAL_SIZE, COLUMNS);
    multiplication_matrix(a, b, check);
    
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
        cout << "Install OpenCl, please" << endl;
        return 0;
    }
    
    vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.size() == 0) {
        cout << "Devices not found." << endl;
        return 0;
    }
    
    cout << "Information about devices:" << endl;
    for (int i = 0; i < devices.size(); ++i) {
        cout << "Device name: " << devices[i].getInfo<CL_DEVICE_NAME>() << endl;
        cout << "Device max compute units: " << devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
        cout << "Device local mem size: " << devices[i].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << endl;
        cout << "Device max work group size: " << devices[i].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl << endl;
    }
    
    cl::Device device = devices[DEVICE_NUMBER];
    cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << endl;
    
    cl::Context context({ device });
    cl::Program program = get_program(device, context, "/Users/mega_user/Desktop/GPU\ /multi_matrix/resources/kernel.cl");
    
    cl::Buffer buffer_a(context, CL_MEM_READ_ONLY, sizeof(float) * ROWS * GENERAL_SIZE);
    cl::Buffer buffer_b(context, CL_MEM_READ_ONLY, sizeof(float) * GENERAL_SIZE * COLUMNS);
    cl::Buffer buffer_res(context, CL_MEM_READ_WRITE, sizeof(float) * ROWS * COLUMNS);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
    
    
    cl_int errcode;
    int R = ROWS;
    int C = COLUMNS;
    int G = GENERAL_SIZE;
    errcode = queue.enqueueWriteBuffer(buffer_a, CL_TRUE, 0, sizeof(float) * ROWS * GENERAL_SIZE, &a[0]);
    errcode |= queue.enqueueWriteBuffer(buffer_b, CL_TRUE, 0, sizeof(float) * GENERAL_SIZE * COLUMNS, &b[0]);
    
    cl::NDRange global(ROWS, COLUMNS);
    cl::NDRange local(BLOCK_SIZE, BLOCK_SIZE);
    cl::Kernel kernel_1 = cl::Kernel(program, "optimization_1_multiplication");
    errcode |= kernel_1.setArg(0, buffer_a);
    errcode |= kernel_1.setArg(1, buffer_b);
    errcode |= kernel_1.setArg(2, buffer_res);
    errcode |= kernel_1.setArg(3, sizeof(int), &R);
    errcode |= kernel_1.setArg(4, sizeof(int), &C);
    errcode |= kernel_1.setArg(5, sizeof(int), &G);
    
    cl::Event event_1;
    errcode |= queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, global,  local, NULL, &event_1);
    errcode |= queue.finish();

    cl_ulong start_time;
    cl_ulong finish_time;
    event_1.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    event_1.getProfilingInfo(CL_PROFILING_COMMAND_END, &finish_time);
    double milliseconds = (finish_time - start_time) / 1000000.0;

    cout << "Execution time in milliseconds: " << milliseconds << endl;

    errcode |= queue.enqueueReadBuffer(buffer_res, CL_TRUE, 0, sizeof(float) * ROWS * COLUMNS, &res[0]);

    if (check_answ(res, check, ROWS, COLUMNS)) {
        cout << "Calculation in " << kernel_1.getInfo<CL_KERNEL_FUNCTION_NAME>() << " is correct." << endl;
    } else {
        cout << "Calculation in " << kernel_1.getInfo<CL_KERNEL_FUNCTION_NAME>() << " is incorrect." << endl;
    }
    
    cl::Kernel kernel_2 = cl::Kernel(program, "simple_multiplication");
    errcode |= kernel_2.setArg(0, buffer_a);
    errcode |= kernel_2.setArg(1, buffer_b);
    errcode |= kernel_2.setArg(2, buffer_res);
    errcode |= kernel_2.setArg(3, sizeof(int), &R);
    errcode |= kernel_2.setArg(4, sizeof(int), &C);

    cl::Event event_2;
    errcode |= queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, global,  cl::NullRange, NULL, &event_2);
    errcode |= queue.finish();

    event_2.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    event_2.getProfilingInfo(CL_PROFILING_COMMAND_END, &finish_time);
    milliseconds = (finish_time - start_time) / 1000000.0;

    cout << "Execution time in milliseconds: " << milliseconds << endl;
    errcode |= queue.enqueueReadBuffer(buffer_res, CL_TRUE, 0, sizeof(float) * ROWS * COLUMNS, &res[0]);
    if (check_answ(res, check, ROWS, COLUMNS)) {
        cout << "Calculation in " << kernel_2.getInfo<CL_KERNEL_FUNCTION_NAME>() << " is correct." << endl;
    } else {
        cout << "Calculation in " << kernel_2.getInfo<CL_KERNEL_FUNCTION_NAME>() << " is incorrect." << endl;
    }
    return 0;
}
