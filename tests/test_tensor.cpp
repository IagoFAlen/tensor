// DEVICE::TENSOR (alloc, feed, fetch), HOST::TENSOR, validate_matmul e status_name.
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

        CHECK_STATUS("feed antes do alloc", t.feed(h), TENSOR_FEED_NOT_ALLOCATED);
        CHECK_STATUS("fetch antes do alloc", t.fetch(back), TENSOR_FETCH_NOT_ALLOCATED);
        CHECK_STATUS("alloc", t.alloc(), TENSOR_SUCCESS);
        CHECK_STATUS("alloc de novo (nao realoca)", t.alloc(), TENSOR_SUCCESS);
        CHECK_STATUS("feed com h_data nulo", t.feed(nullptr), TENSOR_FEED_NULL_DATA);
        CHECK_STATUS("fetch com h_data nulo", t.fetch(nullptr), TENSOR_FETCH_NULL_DATA);
        CHECK_STATUS("feed", t.feed(h), TENSOR_SUCCESS);
        CHECK_STATUS("fetch", t.fetch(back), TENSOR_SUCCESS);

        bool same = true;
        for (int i = 0; i < 6; i++) if (back[i] != h[i]) same = false;
        CHECK("fetch devolve o que o feed mandou", same);
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
        CHECK_STATUS("fetch sincroniza antes de copiar", t.fetch(back), TENSOR_EXEC_FAILED);
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
