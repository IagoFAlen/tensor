/*
 * mat_mul e hadamard_mul de ponta a ponta: TENSOR -> alloc -> feed -> to_tensor3
 * (-> transpose) -> operação -> fetch, conferindo contra a mesma conta feita na CPU.
 * Os valores são inteiros pequenos, então a soma em float é exata e dá para comparar com ==.
 */
#include <initializer_list>
#include <vector>
#include "check.hpp"
#include "device/layout.hpp"
#include "device/ops.hpp"

using namespace DEVICE;

// shape e stride precisam viver enquanto o TENSOR existir: o TENSOR só guarda os ponteiros.
struct Dims {
    size_t sh[8] = {}, st[8] = {}, r = 0;
    Dims(std::initializer_list<size_t> shape, std::initializer_list<size_t> stride){
        for (size_t v : shape)  sh[r++] = v;
        size_t i = 0;
        for (size_t v : stride) st[i++] = v;
    }
};

// Um tensor de teste: os valores no host (h), a cópia na GPU (dev) e a visão 3D (t3).
struct Tensor : Dims {
    TENSOR dev;
    std::vector<float> h;
    TENSOR3 t3 = {};
    bool ready = false;

    Tensor(std::initializer_list<size_t> shape, std::initializer_list<size_t> stride, int seed)
        : Dims(shape, stride), dev(sh, st, r), h(dev.n) {
        for (size_t i = 0; i < h.size(); i++)
            h[i] = seed == 0 ? 0.0f : (float)((i * seed) % 11) - 5.0f;
        ready = dev.alloc() == TENSOR_SUCCESS && dev.feed(h.data()) == TENSOR_SUCCESS && to_tensor3(&dev, &t3) == TENSOR_SUCCESS;
    }

    // Valor lógico [d][r][c] visto por v (que pode ser t3 ou uma transposta dele), lido da cópia no host.
    float at(const TENSOR3& v, size_t d, size_t row, size_t col) const {
        return h[d * v.stride[0] + row * v.stride[1] + col * v.stride[2]];
    }

    std::vector<float> fetched(Status* s) const {
        std::vector<float> out(dev.n);
        *s = dev.fetch(out.data());
        return out;
    }
};

static float out_at(const std::vector<float>& out, const TENSOR3& c, size_t d, size_t row, size_t col){
    return out[d * c.stride[0] + row * c.stride[1] + col * c.stride[2]];
}

// Roda C = A · B e compara com a soma na CPU.
static void matmul_case(const char* name, const Tensor& A, const TENSOR3& a, const Tensor& B, const TENSOR3& b, Tensor& C){
    if (!A.ready || !B.ready || !C.ready) { CHECK(name, false); return; }
    Status s = mat_mul(&a, &b, &C.t3);
    Status f;
    std::vector<float> out = C.fetched(&f);
    bool ok = s == TENSOR_SUCCESS && f == TENSOR_SUCCESS;
    for (size_t d = 0; ok && d < C.t3.shape[0]; d++)
        for (size_t row = 0; row < C.t3.shape[1]; row++)
            for (size_t col = 0; col < C.t3.shape[2]; col++) {
                float sum = 0.0f;
                for (size_t k = 0; k < a.shape[2]; k++) sum += A.at(a, d, row, k) * B.at(b, d, k, col);
                if (out_at(out, C.t3, d, row, col) != sum) ok = false;
            }
    CHECK(name, ok);
}

// Roda C = A ⊙ B e compara elemento a elemento.
static void hadamard_case(const char* name, const Tensor& A, const TENSOR3& a, const Tensor& B, const TENSOR3& b, Tensor& C){
    if (!A.ready || !B.ready || !C.ready) { CHECK(name, false); return; }
    Status s = hadamard_mul(&a, &b, &C.t3);
    Status f;
    std::vector<float> out = C.fetched(&f);
    bool ok = s == TENSOR_SUCCESS && f == TENSOR_SUCCESS;
    for (size_t d = 0; ok && d < C.t3.shape[0]; d++)
        for (size_t row = 0; row < C.t3.shape[1]; row++)
            for (size_t col = 0; col < C.t3.shape[2]; col++)
                if (out_at(out, C.t3, d, row, col) != A.at(a, d, row, col) * B.at(b, d, row, col)) ok = false;
    CHECK(name, ok);
}

int main(){
    SECTION("mat_mul");
    {
        Tensor A({2, 3, 4}, {12, 4, 1}, 7), B({2, 4, 5}, {20, 5, 1}, 3), C({2, 3, 5}, {15, 5, 1}, 0);
        matmul_case("[2, 3, 4] . [2, 4, 5] = [2, 3, 5]", A, A.t3, B, B.t3, C);
    }
    {
        Tensor A({2, 17, 5}, {85, 5, 1}, 7), B({2, 5, 33}, {165, 33, 1}, 3), C({2, 17, 33}, {561, 33, 1}, 0);
        matmul_case("bordas do grid: [2, 17, 5] . [2, 5, 33]", A, A.t3, B, B.t3, C);
    }
    {
        Tensor W({3, 4}, {4, 1}, 7), X({2, 4, 5}, {20, 5, 1}, 3), Z({2, 3, 5}, {15, 5, 1}, 0);
        CHECK("W rank 2 -> [1, 3, 4] com stride[0] = 0", W.t3.shape[0] == 1 && W.t3.stride[0] == 0);
        matmul_case("W compartilhado: [3, 4] . [2, 4, 5]", W, W.t3, X, X.t3, Z);
    }
    {
        // X rank 4 [2, 2, 3, 4] -> [4, 3, 4] -> transposta [4, 4, 3]; Z = X^T . W
        Tensor X({2, 2, 3, 4}, {24, 12, 4, 1}, 7), W({3, 5}, {5, 1}, 3), Z({4, 4, 5}, {20, 5, 1}, 0);
        TENSOR3 xT;
        CHECK_STATUS("transpose(X)", transpose(&X.t3, &xT), TENSOR_SUCCESS);
        matmul_case("X rank 4 transposta: X^T . W = [4, 4, 5]", X, xT, W, W.t3, Z);
    }

    SECTION("hadamard_mul");
    {
        Tensor A({2, 3, 4}, {12, 4, 1}, 7), B({2, 3, 4}, {12, 4, 1}, 3), C({2, 3, 4}, {12, 4, 1}, 0);
        hadamard_case("[2, 3, 4] . [2, 3, 4]", A, A.t3, B, B.t3, C);
    }
    {
        Tensor A({3, 17, 33}, {561, 33, 1}, 7), B({3, 17, 33}, {561, 33, 1}, 3), C({3, 17, 33}, {561, 33, 1}, 0);
        hadamard_case("bordas do grid: [3, 17, 33]", A, A.t3, B, B.t3, C);
    }
    {
        Tensor A({2, 3, 4}, {12, 4, 1}, 7), bias({3, 4}, {4, 1}, 3), C({2, 3, 4}, {12, 4, 1}, 0);
        hadamard_case("bias [3, 4] com broadcast no batch", A, A.t3, bias, bias.t3, C);
    }
    {
        // A guardado como [2, 4, 3] e usado transposto como [2, 3, 4]
        Tensor A({2, 4, 3}, {12, 3, 1}, 7), B({2, 3, 4}, {12, 4, 1}, 3), C({2, 3, 4}, {12, 4, 1}, 0);
        TENSOR3 aT;
        transpose(&A.t3, &aT);
        hadamard_case("A transposta: [2, 4, 3]^T . [2, 3, 4]", A, aT, B, B.t3, C);
    }

    SECTION("validacoes");
    {
        Tensor A({2, 3, 4}, {12, 4, 1}, 7), B35({2, 3, 5}, {15, 5, 1}, 3), B45({2, 4, 5}, {20, 5, 1}, 3);
        Tensor C35({2, 3, 5}, {15, 5, 1}, 0), C53({2, 5, 3}, {15, 3, 1}, 0), C34({2, 3, 4}, {12, 4, 1}, 0);
        Tensor A1({1, 3, 4}, {12, 4, 1}, 7), A3({3, 3, 4}, {12, 4, 1}, 7), B43({2, 4, 3}, {12, 3, 1}, 3);

        CHECK_STATUS("mat_mul: K diferente ([.., 3, 4] . [.., 3, 5])", mat_mul(&A.t3, &B35.t3, &C35.t3), TENSOR_MATMUL_K_MISMATCH);
        CHECK_STATUS("mat_mul: C [2, 5, 3] em vez de [2, 3, 5]", mat_mul(&A.t3, &B45.t3, &C53.t3), TENSOR_MATMUL_C_SHAPE);
        CHECK_STATUS("mat_mul: A [1, 3, 4] com stride[0] = 12", mat_mul(&A1.t3, &B45.t3, &C35.t3), TENSOR_MATMUL_BATCH_MISMATCH);
        CHECK_STATUS("mat_mul: C nulo", mat_mul(&A.t3, &B45.t3, nullptr), TENSOR_MATMUL_NULL_PARAM);

        CHECK_STATUS("hadamard_mul: B [2, 4, 3] com C [2, 3, 4]", hadamard_mul(&A.t3, &B43.t3, &C34.t3), TENSOR_HADAMARD_SHAPE_MISMATCH);
        CHECK_STATUS("hadamard_mul: A batch 3 com C batch 2", hadamard_mul(&A3.t3, &A.t3, &C34.t3), TENSOR_HADAMARD_BATCH_MISMATCH);
        CHECK_STATUS("hadamard_mul: A nulo", hadamard_mul(nullptr, &A.t3, &C34.t3), TENSOR_HADAMARD_NULL_PARAM);

        TENSOR3 sem_memoria = A.t3;
        sem_memoria.data = nullptr;
        CHECK_STATUS("mat_mul: A sem alloc (data nulo)", mat_mul(&sem_memoria, &B45.t3, &C35.t3), TENSOR_MATMUL_NULL_PARAM);
        CHECK_STATUS("hadamard_mul: B sem alloc (data nulo)", hadamard_mul(&A.t3, &sem_memoria, &C34.t3), TENSOR_HADAMARD_NULL_PARAM);
    }

    SECTION("synchronize");
    CHECK_STATUS("depois de todos os kernels", synchronize(), TENSOR_SUCCESS);

    return finish();
}
