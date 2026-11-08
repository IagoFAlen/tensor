# O que falta

Ordem sugerida: de cima para baixo. As seções 4 a 6 vêm conforme a rede neural (no outro projeto) for pedindo.

## 1. Ler o resultado e pegar erros

- [x] **`device_to_host` no `DEVICE::TENSOR`**: copia `data` da GPU para o host.
- [x] **Erro durante a execução do kernel**: `device_to_host` sincroniza antes de copiar, e `DEVICE::synchronize()` sincroniza sem copiar. Os dois devolvem `TENSOR_EXEC_FAILED`.

## 2. Build e teste na GPU

- [ ] Instalar o CUDA Toolkit (`nvcc`).
- [x] `CMakeLists.txt` compilando `src/` como biblioteca, com modo de emulação na CPU (`-DTENSOR_EMULATE=ON`). Ver [build.md](build.md).
- [x] Pasta `tests/` com os testes de layout, operações e tensor.
- [ ] Rodar os testes na GPU (`cmake -S . -B build` com o toolkit instalado). Até agora só rodaram emulados: **o caminho com `nvcc` nunca foi compilado**.

## 3. Pendências pequenas

- [x] Commit das mudanças.
- [x] Docs atualizadas: [tensor.md](tensor.md) (ciclo de vida, cópia proibida) e [row-major.md](row-major.md).
- [x] `validate_matmul` mantido como validação no `TENSOR` de rank N, antes da conversão. É redundante com as checagens do `mat_mul` no `TENSOR3`.
- [x] Comentário de M, K e N de volta acima do `mat_mul`.
- [x] [config/device.hpp](../config/device.hpp) guarda o tamanho do bloco (`BLOCK_X`, `BLOCK_Y`).

## 4. Operações elemento a elemento

O `hadamard_mul` é o molde: mesmo grid, mesmo `batch_fits`, broadcast por stride 0.

- [ ] Soma e subtração (soma cobre o bias com broadcast `[1, M, N]`).
- [ ] Multiplicação por escalar.
- [ ] Aplicar uma função a cada elemento.
- [ ] Avaliar generalizar o `hadamard_mul` em vez de um kernel por operação.

## 5. Outros kernels

- [ ] **Redução**: soma ao longo de um eixo e máximo. É outro padrão de kernel: threads cooperam por shared memory e `__syncthreads()`. A emulação atual não cobre isso.
- [ ] **Cópia contígua**: lê pelo stride e escreve em ordem. Tira o `TENSOR_BATCH_NOT_CONTIGUOUS` do beco sem saída (o "plano B" do `reshape` do PyTorch) e materializa uma transposta quando precisar.
- [ ] **Preencher**: zeros, valor fixo e aleatório (cuRAND, que vem no CUDA Toolkit, ou um gerador próprio).

## 6. Mais tarde

- [x] **`shape` e `stride` próprios no `TENSOR`**: `alloc_shape` e `alloc_stride` copiam os dois (com `malloc`, do tamanho do `rank`), cada um com o seu `Status`. Antes eram ponteiros emprestados no construtor, e um `DEVICE::TENSOR d(h.shape, h.stride, ...)` apontava para memória liberada depois que `h` era destruído (achado no csv2tensor).
- [ ] **Limite do eixo z**: `gridDim.z` vai até 65535. Batch maior que isso faz o lançamento falhar (`TENSOR_LAUNCH_FAILED`).
- [ ] **Desempenho da `mat_mul`**: a versão atual lê cada elemento da memória global K vezes. A versão com *tiling* em shared memory é bem mais rápida.

## Feito antes

- [x] Layout row-major, `contiguous_stride` compartilhado.
- [x] `TENSOR3` e `to_tensor3` (rank ≥ 2, batch fundido, checagem de contiguidade).
- [x] `transpose` no `TENSOR3` (sem copiar dados).
- [x] `mat_mul` e `hadamard_mul` com grid 3D (`dim3`), strides e broadcast por stride 0.
- [x] Um status por verificação, inclusive ponteiros nulos e os `alloc_*`.
- [x] Cópia de `TENSOR` proibida (`= delete`), sem double free.
