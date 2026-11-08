// DEVICE::TENSOR (alloc, feed, host_to_device, device_to_host), HOST::TENSOR, validate_matmul e status_name.
#include <string>
#include <type_traits>
#include "check.hpp"
#include "device/tensor.hpp"
#include "device/ops.hpp"
#include "host/tensor.hpp"
#include "host/layout.hpp"

// Copiar um TENSOR liberaria a mesma memória duas vezes: precisa ser erro de compilação.
static_assert(!std::is_copy_constructible<DEVICE::TENSOR>::value, "DEVICE::TENSOR nao pode ser copiavel");
static_assert(!std::is_copy_assignable<DEVICE::TENSOR>::value,    "DEVICE::TENSOR nao pode ser copiavel");
static_assert(!std::is_copy_constructible<HOST::TENSOR>::value,   "HOST::TENSOR nao pode ser copiavel");
static_assert(!std::is_copy_assignable<HOST::TENSOR>::value,      "HOST::TENSOR nao pode ser copiavel");
// O TENSOR3 não é dono de nada: copiar é seguro (e é assim que ele vai para o kernel).
static_assert(std::is_trivially_copyable<DEVICE::TENSOR3>::value, "TENSOR3 precisa ser copiavel");

int main(){
    SECTION("DEVICE::TENSOR");
    {
        size_t sh[] = {2, 3}, st[] = {3, 1};
        float h[6] = {1, 2, 3, 4, 5, 6}, back[6] = {};
        DEVICE::TENSOR t(sh, st, 2);

        CHECK_STATUS("host_to_device antes do alloc", t.host_to_device(h), TENSOR_HOST_TO_DEVICE_NOT_ALLOCATED);
        CHECK_STATUS("device_to_host antes do alloc", t.device_to_host(back), TENSOR_DEVICE_TO_HOST_NOT_ALLOCATED);
        CHECK_STATUS("alloc", t.alloc(), TENSOR_SUCCESS);
        CHECK_STATUS("alloc de novo (nao realoca)", t.alloc(), TENSOR_SUCCESS);
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
        HOST::TENSOR h(sh, 3);
        DEVICE::TENSOR d(sh, st, 3);

        CHECK_STATUS("GPU sem alloc", d.feed(&h), TENSOR_FEED_NOT_ALLOCATED);
        d.alloc();
        CHECK_STATUS("HOST::TENSOR nulo", d.feed(nullptr), TENSOR_FEED_NULL_DATA);
        CHECK_STATUS("HOST::TENSOR sem alloc", d.feed(&h), TENSOR_FEED_NULL_DATA);

        h.alloc();
        for (size_t i = 0; i < h.n; i++) h.data[i] = (float)i * 0.5f;
        CHECK_STATUS("mesmo shape e stride", d.feed(&h), TENSOR_SUCCESS);

        float back[24] = {};
        d.device_to_host(back);
        bool same = true;
        for (size_t i = 0; i < 24; i++) if (back[i] != h.data[i]) same = false;
        CHECK("device_to_host devolve os dados do HOST::TENSOR", same);

        size_t sh_r2[] = {6, 4}, st_r2[] = {4, 1};
        DEVICE::TENSOR r2(sh_r2, st_r2, 2);
        r2.alloc();
        CHECK_STATUS("rank diferente ([2, 3, 4] -> [6, 4])", r2.feed(&h), TENSOR_FEED_SHAPE_MISMATCH);

        size_t sh_o[] = {2, 4, 3}, st_o[] = {12, 3, 1};
        DEVICE::TENSOR other(sh_o, st_o, 3);
        other.alloc();
        CHECK_STATUS("mesmo n, shape diferente ([2, 3, 4] -> [2, 4, 3])", other.feed(&h), TENSOR_FEED_SHAPE_MISMATCH);

        size_t st_t[] = {12, 1, 3};
        DEVICE::TENSOR transposed(sh, st_t, 3);
        transposed.alloc();
        CHECK_STATUS("mesmo shape, stride transposto [12, 1, 3]", transposed.feed(&h), TENSOR_FEED_STRIDE_MISMATCH);
    }

#ifdef TENSOR_EMULATE
    SECTION("erro dentro de um kernel (simulado, so na emulacao)");
    {
        size_t sh[] = {2}, st[] = {1};
        float back[2];
        DEVICE::TENSOR t(sh, st, 1);
        t.alloc();
        emulate_pending_kernel_error() = cudaErrorUnknown;
        CHECK_STATUS("synchronize", DEVICE::synchronize(), TENSOR_EXEC_FAILED);
        emulate_pending_kernel_error() = cudaErrorUnknown;
        CHECK_STATUS("device_to_host sincroniza antes de copiar", t.device_to_host(back), TENSOR_EXEC_FAILED);
    }
#endif

    SECTION("HOST::TENSOR");
    {
        size_t sh[] = {2, 3, 4};
        HOST::TENSOR t(sh, 3);
        CHECK("antes do alloc: data e stride nulos", t.data == nullptr && t.stride == nullptr);
        CHECK_STATUS("alloc", t.alloc(), TENSOR_SUCCESS);
        CHECK("stride contiguo = [12, 4, 1]", t.stride[0] == 12 && t.stride[1] == 4 && t.stride[2] == 1);
        CHECK("n = 24", t.n == 24);
    }

    SECTION("validate_matmul");
    {
        size_t sh_v[] = {4}, sh_m[] = {4, 3}, sh_n[] = {3, 5};
        HOST::TENSOR v(sh_v, 1), m(sh_m, 2), n(sh_n, 2);
        v.alloc(); m.alloc(); n.alloc();
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
