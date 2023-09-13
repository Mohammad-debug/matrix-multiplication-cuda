#include <cuda_runtime_api.h>
#include<time.h>
#include <iostream>
#include <fstream>
#define   N  512// matrix size
using namespace std;

void allocate_array_2d(double**& pDouble, const int dim1, const int dim2) {
    // Contiguous allocation of 2D arrays

    pDouble = new double * [dim1];
    pDouble[0] = new double[dim1 * dim2];
    for (int i = 1; i < dim1; i++) pDouble[i] = pDouble[i - 1] + dim2;

    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            pDouble[i][j] = 0;
        }
    }
}

void copy_HTD_2d(double**& , double**& pDouble, const int dim1, const int dim2) {
    // Contiguous allocation of 2D arrays

    pDouble = new double* [dim1];
    pDouble[0] = new double[dim1 * dim2];
    for (int i = 1; i < dim1; i++) pDouble[i] = pDouble[i - 1] + dim2;

    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            pDouble[i][j] = 0;
        }
    }
}

// Error checking macro
#define cudaCheckError(code)                                             \
  {                                                                      \
    if ((code) != cudaSuccess) {                                         \
      fprintf(stderr, "Cuda failure %s:%d: '%s' \n", __FILE__, __LINE__, \
              cudaGetErrorString(code));                                 \
    }                                                                    \
  }


__global__ void gpu_matrix_mul(double* da, double* db, double* dc)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    if (x < N && y < N) {  //to protect unauthorised memory access
        double Accumlated_value = 0;
       

        for (int k = 0; k < N; ++k) {

            double elementA = da[y * N + k];  //same as da[y][k]
            double elementB = db[k * N + x];
            Accumlated_value += (elementA * elementB);
        }

        dc[y * N + x] = Accumlated_value;

    }

}


int main()

{
    cout << "   Matrix Multilplication   " << endl;
    cout<< "   _____________________   " << endl << endl;

    ofstream myfile;
    clock_t start_input, end_input, startCPU, endCPU, startGPU, endGPU, start_output, end_output;
    double** a;
    double** b;
    double** c;
    
    allocate_array_2d(a, N, N);//input matrix
    allocate_array_2d(b, N, N);//input matrix
    allocate_array_2d(c, N, N);//output matrix 
   // std::cout.precision(5);
    

    start_input = clock();

    // ......INPUT...........

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = rand();
            b[i][j] = rand();

        }
    }

    end_input = clock();
    double input_time = ((double)(end_input - start_input)) / CLOCKS_PER_SEC;
    cout << "Input Time = " << input_time << "sec \n";

    // ......INPUT END...........


        //     **CPU SECTION START**

        startCPU = clock();
        for (int i = 0; i < N; i++) {

            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        endCPU = clock();

        double cpu_time = ((double)(endCPU - startCPU)) / CLOCKS_PER_SEC;
        cout << "CPU Execution Time (Parallelizable Section) = "<<cpu_time << "sec \n";

        //     **CPU SECTION END**

        //-------------------------------------------------------------

         //     **GPU SECTION STARTs**
        startGPU = clock();
        double* da;
        double* db;
        double* dc;
        int total_elements = N * N;

        cudaCheckError(cudaMalloc(&da, total_elements * sizeof(double)));
        cudaCheckError(cudaMalloc(&db, total_elements * sizeof(double)));
        cudaCheckError(cudaMalloc(&dc, total_elements * sizeof(double)));

        cudaCheckError(cudaMemcpy(da, a[0], total_elements * sizeof(double), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(db, b[0], total_elements * sizeof(double), cudaMemcpyHostToDevice));



        cudaCheckError(cudaMemcpy(dc, c[0], total_elements * sizeof(double), cudaMemcpyHostToDevice));
        //    //  Grid Configration.........................
        dim3 thread(32, 32);
        dim3 block((N + 31) / 32, (N + 31) / 32);
        gpu_matrix_mul << <block, thread >> > (da, db, dc);  //kernel launch
        cudaCheckError(cudaDeviceSynchronize());
        
        cudaCheckError(cudaMemcpy(c[0], dc, total_elements * sizeof(double), cudaMemcpyDeviceToHost));//copy result back to host 

        endGPU = clock();
        double gpu_time = ((double)(endGPU - startGPU)) / CLOCKS_PER_SEC;
        cout << "GPU Execution Time (Parallelizable Section) = " << gpu_time << "sec \n";

        //     **GPU SECTION ENDs**
     
        //---------------------------------------------------------------------------------------------------

                          // ......OUTPUT...........

    start_output = clock();

    myfile.open("output.txt");
    for (int i = 0; i < N; i++) {

        for (int j = 0; j < N; j++) {
            myfile << c[i][j] << ",";
            
        }
        myfile << std::endl;
    }
    myfile.close();


    end_output = clock();
    double output_time = ((double)( end_output-start_output)) / CLOCKS_PER_SEC;
    cout << "OUTPUT (File Write OP) Execution Time = " << output_time << "sec"<<endl<<endl;
                      
                                      // RESULT

    double total_cpu_time = output_time + input_time + cpu_time;
    double total_gpu_time = output_time + input_time + gpu_time;

    cout << "   Theoritical Performance Analysis using amdahl's law "<<endl;
    cout << "   ____________________________________________________ " << endl<<endl;

    cout << "Total CPU Program Time = " << total_cpu_time << "sec \n";
    cout << "Execution Time Parallelizable = " <<  cpu_time << "sec"<<endl;// total CPU Program time - input - output
    //         According to amdhals law:-
    double p = cpu_time / total_cpu_time;
    double pfx = 1 / (1 - p);
    cout << "Theoritical Performance Increase = " << pfx << "x "<<endl<<endl;

    cout << "   Actual Performance Analysis  " << endl;
    cout << "   _____________________________ " << endl << endl;

    cout << "Total GPU Time = " << total_gpu_time << "sec "<<endl;
    cout << "Total CPU Time = " << total_cpu_time << "sec"<<endl;
    double actual_pfx = total_cpu_time / total_gpu_time;
    cout << "Actual Performance Increase =  = " << actual_pfx << "x "<<endl;
    std::cout << "\n**END** " << "\n";
                    

                        //RESULT WRITE IN TEXTFILE

    myfile.open("results.txt");
    myfile << "   Matrix Multilplication   " << endl;
    myfile << "   _____________________   " << endl<<endl;
    myfile << "Input Time = " << input_time << "sec \n";
    myfile << "CPU Execution Time (Parallelizable Section) = " << cpu_time << "sec \n";
    myfile << "GPU Execution Time (Parallelizable Section) = " << gpu_time << "sec \n";
    myfile << "OUTPUT (File Write OP) Execution Time = " << output_time << "sec" << endl << endl;

    myfile << "   Theoritical Performance Analysis using amdahl's law " << endl;
    myfile << "   ____________________________________________________ " << endl << endl;

    myfile << "Total CPU Program Time = " << total_cpu_time << "sec \n";
    myfile << "Execution Time Parallelizable = " << cpu_time << "sec" << endl;// total CPU Program time - input - output
    //         According to amdhals law:-
    myfile << "Theoritical Performance Increase = " << pfx << "x " << endl << endl;

    myfile << "   Actual Performance Analysis  " << endl;
    myfile << "   _____________________________ " << endl << endl;

    myfile << "Total GPU Time = " << total_gpu_time << "sec " << endl;
    myfile << "Total CPU Time = " << total_cpu_time << "sec" << endl;
    myfile << "Actual Performance Increase  = " << actual_pfx << "x " << endl;
    myfile << "\n**END** " << "\n";

    myfile.close();

    //Free device memory
    cudaCheckError( cudaFree(da));
    cudaCheckError( cudaFree(db));
    cudaCheckError( cudaFree(dc));

}

