// DEVICE::TENSOR (alloc_*, feed, host_to_device, device_to_host), HOST::TENSOR, validate_matmul e status_name.
#include <string>
#include <type_traits>
#include "check.hpp"
#include "tensor/device/tensor.hpp"
#include "tensor/device/ops.hpp"
#include "tensor/device/layout.hpp"
#include "tensor/host/tensor.hpp"
#include "tensor/host/layout.hpp"

// Copiar um TENSOR liberaria a mesma memória duas vezes: precisa ser erro de compilação.
static_assert(!std::is_copy_constructible<DEVICE::TENSOR>::value, "DEVICE::TENSOR nao pode ser copiavel");
static_assert(!std::is_copy_assignable<DEVICE::TENSOR>::value,    "DEVICE::TENSOR nao pode ser copiavel");
static_assert(!std::is_copy_constructible<HOST::TENSOR>::value,   "HOST::TENSOR nao pode ser copiavel");
static_assert(!std::is_copy_assignable<HOST::TENSOR>::value,      "HOST::TENSOR nao pode ser copiavel");
// O TENSOR3 não é dono de nada: copiar é seguro (e é assim que ele vai para o kernel).
static_assert(std::is_trivially_copyable<DEVICE::TENSOR3>::value, "TENSOR3 precisa ser copiavel");

// shape + stride + data, parando no primeiro erro.
static Status make_device(DEVICE::TENSOR* t, const size_t* sh, const size_t* st){
    Status s = t->alloc_shape(sh);
    if (s == TENSOR_SUCCESS) s = t->alloc_stride(st);
    if (s == TENSOR_SUCCESS) s = t->alloc_data();
    return s;
}

static Status make_host(HOST::TENSOR* t, const size_t* sh){
    Status s = t->alloc_shape(sh);
    if (s == TENSOR_SUCCESS) s = t->alloc_data();
    return s;
}

int main(){
    SECTION("DEVICE::TENSOR alloc_shape / alloc_stride / alloc_data");
    {
        size_t sh[] = {2, 3}, st[] = {3, 1};
        DEVICE::TENSOR t(2);
        CHECK("construtor: shape, stride e data nulos", t.shape == nullptr && t.stride == nullptr && t.data == nullptr);
        CHECK_STATUS("alloc_data antes do shape", t.alloc_data(), TENSOR_DATA_SHAPE_NOT_ALLOCATED);
        CHECK_STATUS("alloc_shape nulo", t.alloc_shape(nullptr), TENSOR_SHAPE_NULL_PARAM);
        CHECK_STATUS("alloc_shape", t.alloc_shape(sh), TENSOR_SUCCESS);
        CHECK("  n = 6", t.n == 6);
        CHECK_STATUS("alloc_shape de novo", t.alloc_shape(sh), TENSOR_SHAPE_ALREADY_ALLOCATED);
        CHECK_STATUS("alloc_stride nulo", t.alloc_stride(nullptr), TENSOR_STRIDE_NULL_PARAM);
        CHECK_STATUS("alloc_stride", t.alloc_stride(st), TENSOR_SUCCESS);
        CHECK_STATUS("alloc_stride de novo", t.alloc_stride(st), TENSOR_STRIDE_ALREADY_ALLOCATED);
        CHECK_STATUS("alloc_data", t.alloc_data(), TENSOR_SUCCESS);
        CHECK_STATUS("alloc_data de novo (nao realoca)", t.alloc_data(), TENSOR_SUCCESS);
    }

    SECTION("DEVICE::TENSOR host_to_device / device_to_host");
    {
        size_t sh[] = {2, 3}, st[] = {3, 1};
        float h[6] = {1, 2, 3, 4, 5, 6}, back[6] = {};
        DEVICE::TENSOR t(2);
        t.alloc_shape(sh);
        t.alloc_stride(st);

        CHECK_STATUS("host_to_device antes do alloc_data", t.host_to_device(h), TENSOR_HOST_TO_DEVICE_NOT_ALLOCATED);
        CHECK_STATUS("device_to_host antes do alloc_data", t.device_to_host(back), TENSOR_DEVICE_TO_HOST_NOT_ALLOCATED);
        t.alloc_data();
        CHECK_STATUS("host_to_device com h_data nulo", t.host_to_device(nullptr), TENSOR_HOST_TO_DEVICE_NULL_DATA);
        CHECK_STATUS("device_to_host com h_data nulo", t.device_to_host(nullptr), TENSOR_DEVICE_TO_HOST_NULL_DATA);
        CHECK_STATUS("host_to_device", t.host_to_device(h), TENSOR_SUCCESS);
        CHECK_STATUS("device_to_host", t.device_to_host(back), TENSOR_SUCCESS);

        bool same = true;
        for (int i = 0; i < 6; i++) if (back[i] != h[i]) same = false;
        CHECK("device_to_host devolve o que o host_to_device mandou", same);
    }

    SECTION("DEVICE::TENSOR feed");
    {
        size_t sh[] = {2, 3, 4}, st[] = {12, 4, 1};
        HOST::TENSOR h(3);
        DEVICE::TENSOR d(3);
        d.alloc_shape(sh);
        d.alloc_stride(st);

        CHECK_STATUS("GPU sem alloc_data", d.feed(&h), TENSOR_FEED_NOT_ALLOCATED);
        d.alloc_data();
        CHECK_STATUS("HOST::TENSOR nulo", d.feed(nullptr), TENSOR_FEED_NULL_DATA);
        CHECK_STATUS("HOST::TENSOR sem alloc_data", d.feed(&h), TENSOR_FEED_NULL_DATA);

        make_host(&h, sh);
        for (size_t i = 0; i < h.n; i++) h.data[i] = (float)i * 0.5f;
        CHECK_STATUS("mesmo shape e stride", d.feed(&h), TENSOR_SUCCESS);

        float back[24] = {};
        d.device_to_host(back);
        bool same = true;
        for (size_t i = 0; i < 24; i++) if (back[i] != h.data[i]) same = false;
        CHECK("device_to_host devolve os dados do HOST::TENSOR", same);

        size_t sh_r2[] = {6, 4}, st_r2[] = {4, 1};
        DEVICE::TENSOR r2(2);
        make_device(&r2, sh_r2, st_r2);
        CHECK_STATUS("rank diferente ([2, 3, 4] -> [6, 4])", r2.feed(&h), TENSOR_FEED_SHAPE_MISMATCH);

        size_t sh_o[] = {2, 4, 3}, st_o[] = {12, 3, 1};
        DEVICE::TENSOR other(3);
        make_device(&other, sh_o, st_o);
        CHECK_STATUS("mesmo n, shape diferente ([2, 3, 4] -> [2, 4, 3])", other.feed(&h), TENSOR_FEED_SHAPE_MISMATCH);

        size_t st_t[] = {12, 1, 3};
        DEVICE::TENSOR transposed(3);
        make_device(&transposed, sh, st_t);
        CHECK_STATUS("mesmo shape, stride transposto [12, 1, 3]", transposed.feed(&h), TENSOR_FEED_STRIDE_MISMATCH);
    }

#ifdef TENSOR_EMULATE
    SECTION("erro dentro de um kernel (simulado, so na emulacao)");
    {
        size_t sh[] = {2}, st[] = {1};
        float back[2];
        DEVICE::TENSOR t(1);
        make_device(&t, sh, st);
        emulate_pending_kernel_error() = cudaErrorUnknown;
        CHECK_STATUS("synchronize", DEVICE::synchronize(), TENSOR_EXEC_FAILED);
        emulate_pending_kernel_error() = cudaErrorUnknown;
        CHECK_STATUS("device_to_host sincroniza antes de copiar", t.device_to_host(back), TENSOR_EXEC_FAILED);
    }
#endif

    SECTION("shape e stride copiados (nao emprestados)");
    {
        size_t sh[] = {2, 3, 4}, st[] = {12, 4, 1};
        DEVICE::TENSOR d(3);
        HOST::TENSOR   h(3);
        d.alloc_shape(sh);
        d.alloc_stride(st);
        h.alloc_shape(sh);
        CHECK("DEVICE: shape e stride sao outros enderecos", d.shape != sh && d.stride != st);
        CHECK("HOST: shape e outro endereco", h.shape != sh);

        sh[0] = 99; st[0] = 99;                         // quem chamou muda os arrays depois
        CHECK("DEVICE: mudar o array original nao muda o tensor", d.shape[0] == 2 && d.stride[0] == 12);
        CHECK("HOST: mudar o array original nao muda o tensor", h.shape[0] == 2);
    }
    {
        // O caso do csv2tensor: d montado com o shape e o stride de h, e h destruído antes de d.
        size_t sh[] = {2, 3, 4};
        HOST::TENSOR* h = new HOST::TENSOR(3);
        make_host(h, sh);
        for (size_t i = 0; i < h->n; i++) h->data[i] = (float)i;

        DEVICE::TENSOR d(h->rank);
        make_device(&d, h->shape, h->stride);
        CHECK_STATUS("feed de h", d.feed(h), TENSOR_SUCCESS);
        delete h;                                       // libera o host: shape e stride de h somem

        DEVICE::TENSOR3 t3 = {};
        CHECK_STATUS("to_tensor3(d) depois de destruir h", DEVICE::to_tensor3(&d, &t3), TENSOR_SUCCESS);
        CHECK("  shape3 [2, 3, 4], stride3 [12, 4, 1]",
              t3.shape[0] == 2 && t3.shape[1] == 3 && t3.shape[2] == 4 &&
              t3.stride[0] == 12 && t3.stride[1] == 4 && t3.stride[2] == 1);

        float back[24] = {};
        CHECK_STATUS("device_to_host depois de destruir h", d.device_to_host(back), TENSOR_SUCCESS);
        CHECK("  dados intactos", back[0] == 0.0f && back[23] == 23.0f);
    }

    SECTION("HOST::TENSOR");
    {
        size_t sh[] = {2, 3, 4};
        HOST::TENSOR t(3);
        CHECK_STATUS("alloc_data antes do shape", t.alloc_data(), TENSOR_DATA_SHAPE_NOT_ALLOCATED);
        CHECK_STATUS("alloc_shape nulo", t.alloc_shape(nullptr), TENSOR_SHAPE_NULL_PARAM);
        CHECK_STATUS("alloc_shape", t.alloc_shape(sh), TENSOR_SUCCESS);
        CHECK_STATUS("alloc_shape de novo", t.alloc_shape(sh), TENSOR_SHAPE_ALREADY_ALLOCATED);
        CHECK("antes do alloc_data: data e stride nulos", t.data == nullptr && t.stride == nullptr);
        CHECK_STATUS("alloc_data", t.alloc_data(), TENSOR_SUCCESS);
        CHECK("stride contiguo = [12, 4, 1]", t.stride[0] == 12 && t.stride[1] == 4 && t.stride[2] == 1);
        CHECK("n = 24", t.n == 24);
    }

    SECTION("validate_matmul");
    {
        size_t sh_v[] = {4}, sh_m[] = {4, 3}, sh_n[] = {3, 5};
        HOST::TENSOR v(1), m(2), n(2);
        make_host(&v, sh_v); make_host(&m, sh_m); make_host(&n, sh_n);
        CHECK_STATUS("A nulo", HOST::validate_matmul(nullptr, &m), TENSOR_MATMUL_NULL_PARAM);
        CHECK_STATUS("B com rank 1", HOST::validate_matmul(&m, &v), TENSOR_RANK_TOO_SMALL);
        CHECK_STATUS("[4, 3] . [4, 3]", HOST::validate_matmul(&m, &m), TENSOR_MATMUL_K_MISMATCH);
        CHECK_STATUS("[4, 3] . [3, 5]", HOST::validate_matmul(&m, &n), TENSOR_SUCCESS);
    }

    SECTION("status_name");
    {
        bool all_named = true;
        for (int s = 0; s < TENSOR_STATUS_COUNT; s++)
            if (std::string(status_name((Status)s)) == "UNRECOGNIZED STATUS") all_named = false;
        CHECK("todo status tem nome", all_named);
        CHECK("TENSOR_STATUS_COUNT nao tem nome", std::string(status_name(TENSOR_STATUS_COUNT)) == "UNRECOGNIZED STATUS");
    }

    return finish();
}
