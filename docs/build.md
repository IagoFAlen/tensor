# Build e testes

O projeto usa CMake (3.24 ou mais novo) e tem dois modos.

## Na GPU (precisa do CUDA Toolkit)

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Compila `src/` como a biblioteca `tensor`, com os `.cu` passando pelo `nvcc`, para a arquitetura da GPU da máquina (`CUDA_ARCHITECTURES native`). Para fixar a arquitetura, passe `-DCMAKE_CUDA_ARCHITECTURES=89` (RTX 40xx, Ada) no primeiro comando.

### No Ubuntu

- **CMake 3.24 ou mais novo.** O do `apt` no Ubuntu 22.04 é o 3.22. Nesse caso, instale por `pip install cmake` ou `snap install cmake --classic`. O Ubuntu 24.04 já vem com o 3.28.
- **CUDA Toolkit** com o `nvcc` no `PATH`, por exemplo `/usr/local/cuda/bin`. Se o CMake não achar o `nvcc`, passe `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.
- O `nvcc` usa o `gcc` como compilador do host, e cada versão do CUDA aceita só algumas versões do `gcc`. Se aparecer `unsupported GNU version`, passe um `gcc` compatível com `-DCMAKE_CUDA_HOST_COMPILER=g++-12`.
- Para ver se o driver enxerga a GPU: `nvidia-smi`.

### No Windows

O `nvcc` usa o compilador do Visual Studio (MSVC), e o CMake gera uma solução com várias configurações. Por isso, o build e os testes precisam de `--config Release` e `-C Release`.

## Sem GPU (emulação na CPU)

```sh
cmake -S . -B build-emu -DTENSOR_EMULATE=ON
cmake --build build-emu
ctest --test-dir build-emu --output-on-failure
```

Compila tudo com o compilador C++ comum, trocando o CUDA pelo [`tests/emulate/cuda_runtime.h`](../tests/emulate/cuda_runtime.h):

- a memória "da GPU" é memória comum, e `cudaMemcpy` é `memcpy`;
- `kernel<<<grid, block>>>(args)` vira `LAUNCH(kernel, grid, block, args)`, que percorre o grid inteiro thread por thread. O CMake faz essa troca numa cópia de `src/device/ops.cu`, gerada em `build-emu/emulate/ops.cpp`.

A emulação confere índices, strides, bordas do grid e os `Status`. Ela **não** emula paralelismo, shared memory nem `__syncthreads()`, então não substitui rodar na GPU.

Para pegar acesso fora da memória e comportamento indefinido, acrescente os sanitizers (GCC ou Clang):

```sh
cmake -S . -B build-emu -DTENSOR_EMULATE=ON -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -g"
```

## Os testes

Os testes usam só a API pública (`alloc`, `feed`, `host_to_device`, `device_to_host`, `to_tensor3`, `transpose`, `mat_mul`, `hadamard_mul`), então o mesmo código roda nos dois modos.

| Arquivo | O que testa |
|---|---|
| [`test_layout.cpp`](../tests/test_layout.cpp) | `to_tensor3` (ranks 2 a 5, transposta, batch permutado) e `transpose` |
| [`test_ops.cpp`](../tests/test_ops.cpp) | `mat_mul` e `hadamard_mul` de ponta a ponta, conferidos contra a conta na CPU, e as validações |
| [`test_tensor.cpp`](../tests/test_tensor.cpp) | `alloc`, `feed`, `host_to_device`, `device_to_host`, cópia proibida, `HOST::TENSOR`, `validate_matmul` e `status_name` |

O caso de erro dentro de um kernel (`TENSOR_EXEC_FAILED`) só é testado na emulação, que consegue simular a falha.
