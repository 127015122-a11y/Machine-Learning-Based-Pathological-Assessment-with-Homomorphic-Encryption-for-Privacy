#define PROFILE
#include "openfhe/pke/openfhe.h"
using namespace lbcrypto;
#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

void print_double_vector_comma_seperated(const vector<double>& data, const string& label){
    cout<<label<<"\n:[";
    for(size_t i=0;i<data.size()-1;i++)
    {
        cout<<setprecision(15)<<data[i]<<", ";
    }
    cout<<setprecision(15)<<data[data.size()-1]<<"]"<<endl;
}

vector<double> read_double_data_from_file(const string& filename){
    vector<double> data;
    ifstream in(filename);
    if(!in)
    {
        cerr<<"could not open file "<<filename<<endl;
        return data;
    }
    double number;
    while(in >> number)
    {
        data.push_back(number);
    }
    in.close();
    return data;
}

void resize_double_vector(vector<double>& data, uint32_t new_size){
    if(new_size < data.size())
    {
        data.resize(new_size);
    }
    else
    {
        for(size_t i=data.size();i<new_size;i++)
        {
            data.push_back(0.0);
        }
    }
}

vector<vector<double>> read_2d_matrix_from_file(const string& filename){
    ifstream file(filename);
    if(!file.is_open())
    {
        cout<<"error opening file "<<filename<<endl;
        return {};
    }
    vector<vector<double>> matrix;
    string line;
    while(getline(file,line))
    {
        vector<double> row;
        stringstream ss(line);
        double value;
        while(ss>>value)
        {
            row.push_back(value);
        }
        matrix.push_back(row);
    }
    file.close();
    return matrix;
}

void print_matrix(vector<vector<double>> matrix,string label){
    cout<<label<<": [\n";
    for(size_t i=0;i<matrix.size();i++)
    {
        cout<<"[";
        for(size_t j=0;j<matrix[i].size();j++)
        {
            cout<<fixed<<setw(18)<<setprecision(12)<<matrix[i][j] << " ";
        }
        cout<<"]"<<endl;
    }
    cout<<"]"<<endl;
}

int main(){
    cout<<" SVM Polynomial Kernel started... !\n\n";
    

    double gamma=2;
    uint32_t degree=3;
    vector<vector<double>> support_vectors=read_2d_matrix_from_file("E:\\svm_poly\\support_vectors_poly_heart.txt");
    cout<<"number of support vectors: "<<support_vectors.size()<<"\n";
    cout<<"dimension of each support vector: "<<support_vectors[0].size()<<"\n";

     uint32_t n=support_vectors[0].size();
     uint32_t batchSize=16;
     
    print_matrix(support_vectors,"support vectors");

    vector<double> dual_coeffs=read_double_data_from_file("E:\\svm_poly\\dual_coeff_poly_heart.txt");
    vector<double> bias=read_double_data_from_file("E:\\svm_poly\\intercept_poly_heart.txt");
    vector<double> x=read_double_data_from_file("E:\\svm_poly\\xtest_poly_heart.txt");

    resize_double_vector(x,batchSize);
    resize_double_vector(bias,batchSize);

    cout<<"******************************************************************************************"<<endl;
    print_double_vector_comma_seperated(dual_coeffs,"dual_coeff");
    cout<<"******************************************************************************************"<<endl;

    print_double_vector_comma_seperated(bias,"bias");
    cout<<"******************************************************************************************"<<endl;
    print_double_vector_comma_seperated(x,"x");
    cout<<"******************************************************************************************"<<endl;

    
    //setup CryptoContext
    uint32_t multDepth=6;
    uint32_t scaleModSize=50;
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    CryptoContext<DCRTPoly> cc=GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cout<<"CKKS scheme is using ring dimension "<<cc->GetRingDimension()<<endl<<endl;

    auto keys=cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalSumKeyGen(keys.secretKey);

    vector<double> zeros(batchSize,0.0);


    Plaintext pt_zeros=cc->MakeCKKSPackedPlaintext(zeros);
    
    vector<double> gamma_vec(batchSize,0.0);
    gamma_vec[0]=gamma;
    Plaintext pt_gamma=cc->MakeCKKSPackedPlaintext(gamma_vec);

    vector<double> kernel_poly_coeffs(degree+1,0.0);
    kernel_poly_coeffs[degree]=1;

    Plaintext pt_x=cc->MakeCKKSPackedPlaintext(x);
    vector<Plaintext> pt_support_vectors;
    for(auto vector:support_vectors)
    {
        pt_support_vectors.push_back(cc->MakeCKKSPackedPlaintext(vector));

    }
    Plaintext pt_bias=cc->MakeCKKSPackedPlaintext(bias);

    auto ct_x=cc->Encrypt(keys.publicKey,pt_x);

    TimeVar t;
    TIC(t);
    
    auto ct_res=cc->Encrypt(keys.publicKey,pt_zeros);
    for(size_t i=0;i<pt_support_vectors.size();i++)
    {
        cout<<"iteration: "<<i+1<<"\n";
        auto dot_prod=cc->EvalInnerProduct(ct_x,pt_support_vectors[i],n);
        auto ct_gamma_dot_prod=cc->EvalMult(dot_prod,pt_gamma);
        auto ct_kernel_out=cc->EvalPoly(ct_gamma_dot_prod,kernel_poly_coeffs);
        auto ct_out=cc->EvalMult(ct_kernel_out,dual_coeffs[i]);
        ct_res+=ct_out;
    }
    ct_res=cc->EvalAdd(ct_res,pt_bias);

    auto timeEvalSVMTime=TOC_MS(t);
    cout<<"SVM inference took: "<<timeEvalSVMTime<<" ms\n\n";

    Plaintext result;
    cout.precision(8);
    cout<<endl<<"Result of homomorphic computations: "<<endl;

    cc->Decrypt(keys.secretKey,ct_res,&result);
    result->SetLength(batchSize);
    cout<<"computed classification score = "<<result<<endl;
    cout<<"Estimated precision in bits : "<<result->GetLogPrecision()<<endl;
    vector<double> resultVector = result->GetRealPackedValue();

    double score = resultVector[0];
    cout<<"decision value:"<<score<<endl;
    
    if(score>=0)
    {
        cout<<"Person is Diseased"<<endl;
    }
    else
    {
       cout<<"Person is not Diseased"<<endl;
    }
    cout<<"SVM Polynomial Kernel terminated gracefully ... !\n";

    return 0;
}   

