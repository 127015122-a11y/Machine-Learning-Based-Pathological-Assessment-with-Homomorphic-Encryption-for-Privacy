#define PROFILE

#include "openfhe.h"

using namespace lbcrypto;

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

void print_double_vector_comma_separated(const vector<double>& data, const string& label) {
    cout << label << ": [";
    for (size_t i = 0; i < data.size() - 1; i++) {
        cout << setprecision(15) << data[i] << ", ";
    }
    cout << setprecision(15) << data[data.size() - 1] << "]" << endl;
}


vector<double> read_double_data_from_file(const string& filename) {
    vector<double> data;
    ifstream in(filename);
    if (!in) {
        cerr << "Could not open file " << filename << endl;
        return data;
    }

    double number;
    while (in >> number) {
        data.push_back(number);
    }

  
    in.close();

    return data;
}

void resize_double_vector(vector<double>& data, uint32_t new_size) {
    if (new_size < data.size()) {
        data.resize(new_size);
    } else {
        for (size_t i = data.size(); i < new_size; i++) {
            data.push_back(0.0);
        }
    }
}

vector<vector<double>> read_2d_matrix_from_file(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error opening file " << filename << endl;
        return {};
    }

     vector<vector<double>> matrix;
    string line;
    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        double value;
        while (ss >> value) {
            row.push_back(value);
        }
        matrix.push_back(row);
    }

    file.close();

    return matrix;
}

void print_matrix(vector<vector<double>> matrix, string label) {
    cout << label << ": [\n";
    for (size_t i = 0; i < matrix.size(); i++) {
        cout << "[";
        for (size_t j = 0; j < matrix[i].size(); j++) {
            cout << setw(15) << setprecision(15) << matrix[i][j] << " ";
        }
        cout << "]" << endl;
    }
    cout << "]" << endl;
}

int main() {

    cout << "SVM Sigmoid Kernel v1 started ... !\n\n";

    uint32_t n = 13; 
    double gamma = 1.0/n;
   
    vector<vector<double>> support_vectors = read_2d_matrix_from_file("E:\\svm_sigmoid\\support_vectors_sigmoid_heart.txt");
    cout << "number of support vectors: " << support_vectors.size() << "\n";
    cout << "dimension of each support vector: " << support_vectors[0].size() << "\n";
    print_matrix(support_vectors, "support vectors");

    vector<double> dual_coeffs = read_double_data_from_file("E:\\svm_sigmoid\\dual_coeff_sigmoid_heart.txt");
    vector<double> bias = read_double_data_from_file("E:\\svm_sigmoid\\intercept_sigmoid_heart.txt");
    resize_double_vector(bias, n);
    vector<double> x = read_double_data_from_file("E:\\svm_sigmoid\\xtest_sigmoid_heart.txt");
    
    print_double_vector_comma_separated(dual_coeffs, "dual_coeff");
    print_double_vector_comma_separated(bias, "bias");
    print_double_vector_comma_separated(x, "x");
  

    uint32_t multDepth = 14; 
    uint32_t scaleModSize = 59;
    uint32_t batchSize = 16;
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);
    
    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << endl << endl;


    auto keys = cc->KeyGen();

    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalSumKeyGen(keys.secretKey);


    vector<double> zeros(batchSize, 0.0);
    Plaintext pt_zeros = cc->MakeCKKSPackedPlaintext(zeros);
   
    vector<double> gamma_vec(batchSize, 0.0);
    gamma_vec[0] = gamma;
    Plaintext pt_gamma = cc->MakeCKKSPackedPlaintext(gamma_vec);
    
    

    Plaintext pt_x = cc->MakeCKKSPackedPlaintext(x);
    vector<Plaintext> pt_support_vectors;
    for (auto vector : support_vectors) {
        pt_support_vectors.push_back(cc->MakeCKKSPackedPlaintext(vector));
    }
    Plaintext pt_bias = cc->MakeCKKSPackedPlaintext(bias);

    auto ct_x = cc->Encrypt(keys.publicKey, pt_x);
    cout << "num levels in input ctxt: " << ct_x->GetLevel() << "\n";
    cout << "num towers in input ctxt: " << ct_x->GetElements()[0].GetAllElements().size() << endl;
    
 
    double lowerBound = -60.0, upperBound = 60.0;
    uint32_t polyDegree = 495; 

#ifdef VERBOSE
    auto DecAndPrint = [cc, batchSize, keys] (lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ctxt,string label)
    {
        Plaintext res;

        cout.precision(8);
        cc->Decrypt(keys.secretKey, ctxt, &res);
        res->SetLength(batchSize);
        cout << label << ": " << res;
        
    };
    #else
    auto DecAndPrint = [cc, batchSize, keys] (lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ctxt,string label)
    {
        
    };
#endif

    TimeVar t;
    TIC(t);
    auto ct_res = cc->Encrypt(keys.publicKey, pt_zeros);
    for (size_t i = 0; i < pt_support_vectors.size(); i++) {
        #ifdef VERBOSE
        cout << "BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN BEGIN\n";
        #endif
        cout << "iteration: " << i+1 << "\n"; 

        auto dot_prod = cc->EvalInnerProduct(ct_x, pt_support_vectors[i], n);
        DecAndPrint(dot_prod, "dot_prod");
        dot_prod = cc->EvalMult(dot_prod, pt_gamma);
        DecAndPrint(dot_prod, "dot_prod*gamma");

        auto ct_out = cc->EvalChebyshevFunction([](double y) -> double { return tanh(y); }, dot_prod, lowerBound,
                                        upperBound, polyDegree);
        DecAndPrint(ct_out, "tanh");
        ct_out = cc->EvalMult(ct_out, dual_coeffs[i]);
        ct_res += ct_out;
        #ifdef VERBOSE
        cout << "END END END END END END END END END END END END END END END END END \n";
        #endif
    }
    ct_res = cc->EvalAdd(ct_res, pt_biasauto timeEvalSVMTime = TOC_MS(t););
    
    std::cout << "Linear-SVM inference took: " << timeEvalSVMTime << " ms\n\n"; 

    Plaintext result;
    cout.precision(8);

    cout << endl << "Results of homomorphic computations: " << endl;
    cout << "num levels in result ctxt: " << ct_res->GetLevel() << "\n";
    cout << "num towers in result ctxt: " << ct_res->GetElements()[0].GetAllElements().size() << endl;

    cc->Decrypt(keys.secretKey, ct_res, &result);
    result->SetLength(batchSize);
    cout << "computed classification score = " << result;
    cout << "Estimated precision in bits: " << result->GetLogPrecision() << endl;
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

    cout << "SVM Sigmoid Kernel terminated gracefully ... !\n";

    return 0;
}