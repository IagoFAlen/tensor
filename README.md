# tensor

Biblioteca de tensores em C++/CUDA, com memória no host (`HOST`) e na GPU (`DEVICE`). É a base para uma rede neural, que fica em outro projeto.

O design segue o espírito do cuDNN, que serve só de referência: descritor de tensor com shape e stride, e operações que validam a entrada e devolvem um status. Nada aqui depende do cuDNN.

> Em desenvolvimento. Hoje: alocação e cópia host ↔ GPU, `mat_mul` e `hadamard_mul` com batch e broadcast, e `transpose` sem cópia. O que falta está em [docs/todo.md](docs/todo.md).

## Modelo

- **Row-major**: `shape` vai da dimensão mais lenta para a mais rápida; `stride` é medido em elementos (`[2, 3, 4]` → `[12, 4, 1]`).
- **`HOST::TENSOR`** e **`DEVICE::TENSOR`** são donos da memória e não podem ser copiados. O construtor só recebe o `rank`; a memória vem de `alloc_shape`, `alloc_stride` e `alloc_data`, cada um devolvendo um `Status`.
- **`DEVICE::TENSOR3`** é uma view `[batch, linhas, colunas]` de um `DEVICE::TENSOR` de rank ≥ 2, sem dono e sem cópia. As operações recebem `TENSOR3`. Rank maior que 3 tem as dimensões de batch fundidas: `[2, 4, 3, 4]` → `[8, 3, 4]`.
- **Broadcast** por stride 0 no batch: `[1, M, N]` com `stride[0] = 0` vale para todo o batch.
- **Erros**: toda função devolve um `Status`, um por verificação (`TENSOR_MATMUL_K_MISMATCH`, `TENSOR_FEED_SHAPE_MISMATCH`...). `status_name()` dá o texto.

Mais em [docs/tensor.md](docs/tensor.md) e [docs/row-major.md](docs/row-major.md).

## Exemplo

```cpp
#include <cstdio>
#include "tensor/device/tensor.hpp"
#include "tensor/device/layout.hpp"
#include "tensor/device/ops.hpp"

int main(){
    // A [2, 3] · B [3, 2] = C [2, 2]
    size_t sh_a[] = {2, 3}, st_a[] = {3, 1};
    size_t sh_b[] = {3, 2}, st_b[] = {2, 1};
    size_t sh_c[] = {2, 2}, st_c[] = {2, 1};

    float a[] = {1, 2, 3,
                 4, 5, 6};
    float b[] = {1, 0,
                 0, 1,
                 1, 1};

    DEVICE::TENSOR A(2), B(2), C(2);       // rank 2
    A.alloc_shape(sh_a); A.alloc_stride(st_a); A.alloc_data();
    B.alloc_shape(sh_b); B.alloc_stride(st_b); B.alloc_data();
    C.alloc_shape(sh_c); C.alloc_stride(st_c); C.alloc_data();
    A.host_to_device(a);
    B.host_to_device(b);

    DEVICE::TENSOR3 a3, b3, c3;            // views [batch, linhas, colunas]
    DEVICE::to_tensor3(&A, &a3);
    DEVICE::to_tensor3(&B, &b3);
    DEVICE::to_tensor3(&C, &c3);

    Status st = DEVICE::mat_mul(&a3, &b3, &c3);
    if(st != TENSOR_SUCCESS){
        std::printf("mat_mul: %s\n", status_name(st));
        return 1;
    }

    float c[4];
    C.device_to_host(c);                   // sincroniza e copia
    std::printf("[%g, %g]\n[%g, %g]\n", c[0], c[1], c[2], c[3]);   // [4, 5] [10, 11]
}
```

O exemplo só confere o status da `mat_mul`; num código de verdade, confira o de cada chamada. `alloc_shape` e `alloc_stride` copiam os valores, então os arrays originais podem sair de escopo depois.

## Build e testes

Precisa de CMake 3.24+, um compilador C++17 e o CUDA Toolkit.

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Sem GPU, os kernels podem ser emulados na CPU: `cmake -S . -B build-emu -DTENSOR_EMULATE=ON`. A emulação confere índices, strides e status, mas não substitui rodar na GPU.

Detalhes, Ubuntu, Windows e sanitizers em [docs/build.md](docs/build.md).

## Instalação

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo cmake --install build          # /usr/local; outro lugar: --prefix /caminho
```

Os headers vão para `/usr/local/include/tensor/`, a biblioteca para `/usr/local/lib/libtensor.a` e o pacote do CMake para `/usr/local/lib/cmake/tensor/`.

```cmake
find_package(tensor 0.1 REQUIRED)
target_link_libraries(app PRIVATE tensor::tensor)
```

Sem CMake: `g++ -std=c++17 -I/usr/local/cuda/include app.cpp -ltensor -lcudart`.

A biblioteca é compilada para a GPU da máquina (`CMAKE_CUDA_ARCHITECTURES=native`). Para outra GPU, passe `-DCMAKE_CUDA_ARCHITECTURES=89` (por exemplo) no primeiro comando.

## Estrutura

```
include/tensor/        API pública
  status.hpp           Status e status_name
  stride.hpp           contiguous_stride
  host/                HOST::TENSOR, validate_matmul
  device/              DEVICE::TENSOR, TENSOR3, to_tensor3, transpose, mat_mul, hadamard_mul
src/                   implementação (.cpp e .cu)
config/device.hpp      tamanho do bloco dos kernels (interno)
tests/                 testes e a emulação do CUDA
docs/                  conceitos, build e pendências
```

## Projetos relacionados

- [csv2tensor](https://github.com/IagoFAlen/csv2tensor): lê CSV (números e categorias em one-hot) e entrega um `DEVICE::TENSOR3`.
