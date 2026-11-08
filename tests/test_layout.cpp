// to_tensor3 e transpose: só mexem em shape e stride, então nenhum caso aqui toca a memória da GPU.
#include "check.hpp"
#include "tensor/device/layout.hpp"

using namespace DEVICE;

static bool eq3(const size_t* a, size_t x, size_t y, size_t z){ return a[0] == x && a[1] == y && a[2] == z; }

// Converte e confere shape3 e stride3.
static void case_ok(const char* name, size_t* sh, size_t* st, size_t r, const size_t want_sh[3], const size_t want_st[3]){
    TENSOR t(sh, st, r);
    TENSOR3 t3 = {};
    Status s = to_tensor3(&t, &t3);
    CHECK(name, s == TENSOR_SUCCESS && eq3(t3.shape, want_sh[0], want_sh[1], want_sh[2]) && eq3(t3.stride, want_st[0], want_st[1], want_st[2]));
}

int main(){
    SECTION("to_tensor3");
    {
        size_t sh2[] = {3, 4},          st2[] = {4, 1};
        size_t sh3[] = {2, 3, 4},       st3[] = {12, 4, 1};
        size_t sh4[] = {2, 4, 3, 4},    st4[] = {48, 12, 4, 1};
        size_t sh5[] = {2, 2, 4, 3, 4}, st5[] = {96, 48, 12, 4, 1};
        size_t shT[] = {2, 4, 4, 3},    stT[] = {48, 12, 1, 4};

        const size_t s2[] = {1, 3, 4},  t2[] = {0, 4, 1};
        const size_t s3[] = {2, 3, 4},  t3[] = {12, 4, 1};
        const size_t s4[] = {8, 3, 4},  t4[] = {12, 4, 1};
        const size_t s5[] = {16, 3, 4}, t5[] = {12, 4, 1};
        const size_t sT[] = {8, 4, 3},  tT[] = {12, 1, 4};

        case_ok("rank 2 [3, 4] -> [1, 3, 4], stride [0, 4, 1]", sh2, st2, 2, s2, t2);
        case_ok("rank 3 [2, 3, 4] -> [2, 3, 4]", sh3, st3, 3, s3, t3);
        case_ok("rank 4 [2, 4, 3, 4] -> [8, 3, 4], stride [12, 4, 1]", sh4, st4, 4, s4, t4);
        case_ok("rank 5 [2, 2, 4, 3, 4] -> [16, 3, 4]", sh5, st5, 5, s5, t5);
        case_ok("rank 4, matrizes transpostas -> stride [12, 1, 4]", shT, stT, 4, sT, tT);
    }
    {
        size_t sh1[] = {4}, st1[] = {1};
        TENSOR t(sh1, st1, 1);
        TENSOR3 t3 = {};
        CHECK_STATUS("rank 1", to_tensor3(&t, &t3), TENSOR_RANK_TOO_SMALL);
    }
    {
        size_t shP[] = {2, 3, 2, 2}, stP[] = {4, 8, 2, 1};
        TENSOR t(shP, stP, 4);
        TENSOR3 t3 = {};
        CHECK_STATUS("batch permutado, stride [4, 8, 2, 1]", to_tensor3(&t, &t3), TENSOR_BATCH_NOT_CONTIGUOUS);
        CHECK("  out intacto depois da falha", eq3(t3.shape, 0, 0, 0) && eq3(t3.stride, 0, 0, 0));
    }
    {
        size_t sh[] = {2, 3}, st[] = {3, 1};
        TENSOR t(sh, st, 2), sem_stride(sh, nullptr, 2);
        TENSOR3 t3 = {};
        CHECK_STATUS("in nulo", to_tensor3(nullptr, &t3), TENSOR_LAYOUT_NULL_PARAM);
        CHECK_STATUS("out nulo", to_tensor3(&t, nullptr), TENSOR_LAYOUT_NULL_PARAM);
        CHECK_STATUS("in->stride nulo", to_tensor3(&sem_stride, &t3), TENSOR_LAYOUT_NULL_PARAM);
    }

    SECTION("transpose");
    {
        float* fake = reinterpret_cast<float*>(0x1000);    // só o endereço importa: transpose não lê data
        TENSOR3 a = {fake, {2, 3, 4}, {12, 4, 1}}, t;
        CHECK_STATUS("[2, 3, 4]", transpose(&a, &t), TENSOR_SUCCESS);
        CHECK("  shape [2, 3, 4] -> [2, 4, 3]", eq3(t.shape, 2, 4, 3));
        CHECK("  stride [12, 4, 1] -> [12, 1, 4]", eq3(t.stride, 12, 1, 4));
        CHECK("  mesma memoria (data igual)", t.data == a.data);
        CHECK("  original intacto", eq3(a.shape, 2, 3, 4) && eq3(a.stride, 12, 4, 1));

        TENSOR3 tt;
        transpose(&t, &tt);
        CHECK("transposta da transposta = original", eq3(tt.shape, 2, 3, 4) && eq3(tt.stride, 12, 4, 1));

        TENSOR3 s = a;
        transpose(&s, &s);
        CHECK("no lugar (in == out)", eq3(s.shape, 2, 4, 3) && eq3(s.stride, 12, 1, 4));

        TENSOR3 b = {fake, {1, 3, 4}, {0, 4, 1}}, bt;
        transpose(&b, &bt);
        CHECK("broadcast: stride[0] = 0 preservado", eq3(bt.shape, 1, 4, 3) && eq3(bt.stride, 0, 1, 4));

        CHECK_STATUS("in nulo", transpose(nullptr, &t), TENSOR_TRANSPOSE_NULL_PARAM);
        CHECK_STATUS("out nulo", transpose(&a, nullptr), TENSOR_TRANSPOSE_NULL_PARAM);
    }

    return finish();
}
