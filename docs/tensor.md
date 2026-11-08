# O que é cada coisa no tensor

Este documento mostra o que cada campo de [`HOST::tensor`](../include/tensor/host/tensor.hpp) e [`DEVICE::tensor`](../include/tensor/device/tensor.hpp) representa, usando o shape `[2, 3, 4]` como exemplo. Para ver como os strides são calculados, leia [row-major.md](row-major.md).

## Visão geral

As duas structs têm os mesmos campos:

| Campo | Tipo | O que é | No exemplo |
|---|---|---|---|
| `rank` | `size_t` | Quantidade de dimensões | `3` |
| `shape` | `size_t*` | Tamanho de cada dimensão | `[2, 3, 4]` |
| `stride` | `size_t*` | Quantos elementos avançar para somar 1 ao índice de cada dimensão | `[12, 4, 1]` |
| `n` | `size_t` | Total de elementos | `24` |
| `data` | `float*` | Início dos valores, guardados em sequência | 24 valores `float` |

```text
                  ┌───────────────┐
 rank = 3         │ quantas       │
                  │ dimensões     │
                  └───────┬───────┘
                          ▼
 shape  = [  2,   3,   4 ]    tamanho de cada dimensão
 stride = [ 12,   4,   1 ]    tamanho do passo em cada dimensão
            ▲    ▲    ▲
            i    j    k       índice usado em cada dimensão

 n = 2 × 3 × 4 = 24           total de elementos

 data ──► [ 0 ][ 1 ][ 2 ] ... [ 23 ]
```

## Dimensões

Cada índice escolhe uma posição em uma dimensão. Neste exemplo, as duas últimas dimensões são as linhas e as colunas, e a primeira escolhe o bloco.

```text
shape = [2, 3, 4]
         │  │  └─ k: 4 colunas por linha
         │  └──── j: 3 linhas por bloco
         └─────── i: 2 blocos
```

```text
         i = 0 (bloco 0)                  i = 1 (bloco 1)

         k=0  k=1  k=2  k=3               k=0  k=1  k=2  k=3
       ┌────┬────┬────┬────┐            ┌────┬────┬────┬────┐
 j = 0 │  0 │  1 │  2 │  3 │      j = 0 │ 12 │ 13 │ 14 │ 15 │
       ├────┼────┼────┼────┤            ├────┼────┼────┼────┤
 j = 1 │  4 │  5 │  6 │  7 │      j = 1 │ 16 │ 17 │ 18 │ 19 │
       ├────┼────┼────┼────┤            ├────┼────┼────┼────┤
 j = 2 │  8 │  9 │ 10 │ 11 │      j = 2 │ 20 │ 21 │ 22 │ 23 │
       └────┴────┴────┴────┘            └────┴────┴────┴────┘
```

Os números nas células são as posições em `data`, não os valores guardados.

## `rank`

O `rank` é a quantidade de dimensões, ou seja, o tamanho dos arrays `shape` e `stride`.

| Rank | Exemplo de shape | Forma |
|---|---|---|
| 1 | `[4]` | Vetor |
| 2 | `[3, 4]` | Matriz de 3 linhas e 4 colunas |
| 3 | `[2, 3, 4]` | 2 matrizes de 3 linhas e 4 colunas |

## `stride`

Cada stride é o tamanho do passo para somar 1 ao índice daquela dimensão:

```text
stride[2] = 1    próxima coluna

  0 → 1 → 2 → 3


stride[1] = 4    próxima linha

  0  1  2  3
  ↓
  4  5  6  7


stride[0] = 12   próximo bloco

  bloco 0: posição 0  ──► bloco 1: posição 12
```

## `n` e `data`

`data` guarda os elementos em uma única sequência. A divisão em blocos, linhas e colunas vem de `shape` e `stride`, não da memória:

```text
data
 │
 ▼
┌───┬───┬───┬───┬───┬───┬───┬───┬─────┬────┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ ... │ 23 │
└───┴───┴───┴───┴───┴───┴───┴───┴─────┴────┘
└──────────────── n = 24 ────────────────┘
```

A posição de um elemento é:

```text
posição = i * stride[0] + j * stride[1] + k * stride[2]

(1, 2, 3) → 1 * 12 + 2 * 4 + 3 * 1 = 23
```

## Onde cada coisa fica

A principal diferença entre as duas structs é onde `data` fica e de onde vem `stride`:

| Campo | `HOST::tensor` | `DEVICE::tensor` |
|---|---|---|
| `data` | Alocado na RAM com `malloc`, no `alloc_data()` | Alocado na GPU com `cudaMalloc`, no `alloc_data()` |
| Valores | Não preenche `data` | `feed` e `host_to_device` copiam do host para a GPU; `device_to_host` copia de volta |
| `shape` | Copiado no `alloc_shape()` | Copiado no `alloc_shape()` |
| `stride` | Calculado no `alloc_data()` | Copiado no `alloc_stride()` |

```text
          RAM                                 GPU

 HOST::tensor
 ├─ data ───► [ valores ]
 ├─ shape ──► [ 2, 3, 4 ]
 └─ stride ─► [ 12, 4, 1 ]
                   │ alloc_shape e alloc_stride copiam os valores
                   ▼
 DEVICE::tensor
 ├─ shape ──► [ 2, 3, 4 ]     cópia própria
 ├─ stride ─► [ 12, 4, 1 ]    cópia própria
 └─ data ─────────────────────────────► [ valores copiados ]
```

O desenho mostra o caso em que `DEVICE::tensor` é montado com o `shape` e o `stride` de um `HOST::tensor`, como em `d.alloc_shape(h.shape)` e `d.alloc_stride(h.stride)`. Os dois copiam os valores para arrays do próprio tensor, então `d` continua válido mesmo depois que `h` é destruído. A struct `DEVICE::tensor` e os arrays `shape` e `stride` ficam na RAM, e só `data` fica na GPU.

## Ciclo de vida

O construtor só guarda o `rank` e não aloca nada, porque um construtor não consegue devolver `Status`. Toda memória vem dos `alloc_*`, cada um com o seu `Status`:

```cpp
size_t shape[]  = {2, 3, 4};
size_t stride[] = {12, 4, 1};

DEVICE::TENSOR t(3);                  // só o rank: shape, stride e data == nullptr
Status s = t.alloc_shape(shape);      // copia o shape e calcula n
s = t.alloc_stride(stride);           // copia o stride
s = t.alloc_data();                   // n floats na GPU
s = t.feed(&h);                       // HOST::TENSOR -> GPU, conferindo shape e stride
// ... operações ...
s = t.device_to_host(h_result);       // GPU -> host; espera os kernels terminarem antes
```

No `HOST::TENSOR` não existe `alloc_stride`: o `alloc_data()` calcula o stride contíguo a partir do shape.

| Método | Devolve, além de `TENSOR_SUCCESS` |
|---|---|
| `alloc_shape(sh)` | `TENSOR_SHAPE_NULL_PARAM`, `TENSOR_SHAPE_ALREADY_ALLOCATED`, `TENSOR_SHAPE_ALLOC_FAILED` |
| `alloc_stride(st)` (device) | `TENSOR_STRIDE_NULL_PARAM`, `TENSOR_STRIDE_ALREADY_ALLOCATED`, `TENSOR_STRIDE_ALLOC_FAILED` |
| `alloc_data()` | `TENSOR_DATA_SHAPE_NOT_ALLOCATED`, `TENSOR_ALLOC_FAILED` (device) ou `TENSOR_HOST_ALLOC_FAILED` (host) |
| `feed(host_tensor)` | `TENSOR_FEED_NOT_ALLOCATED`, `TENSOR_FEED_NULL_DATA`, `TENSOR_FEED_SHAPE_MISMATCH`, `TENSOR_FEED_STRIDE_MISMATCH` e, da cópia, `TENSOR_HOST_TO_DEVICE_COPY_FAILED` |
| `host_to_device(h_data)` | `TENSOR_HOST_TO_DEVICE_NOT_ALLOCATED`, `TENSOR_HOST_TO_DEVICE_NULL_DATA`, `TENSOR_HOST_TO_DEVICE_COPY_FAILED` |
| `device_to_host(h_data)` | `TENSOR_DEVICE_TO_HOST_NOT_ALLOCATED`, `TENSOR_DEVICE_TO_HOST_NULL_DATA`, `TENSOR_EXEC_FAILED`, `TENSOR_DEVICE_TO_HOST_COPY_FAILED` |

O `host_to_device(const float*)` confia que o buffer tem `n` floats no layout do tensor da GPU. O `feed(const HOST::TENSOR*)` confere isso antes de copiar: mesmo rank, mesmo shape e mesmo stride. O stride precisa bater porque a cópia é byte a byte. Como o `HOST::TENSOR` é sempre contíguo, um tensor da GPU com stride transposto recusa o feed. Para trabalhar com a transposta, faça o feed no tensor contíguo e use `transpose` no `TENSOR3`.


O `alloc_shape` e o `alloc_stride` copiam os valores com `malloc`, do tamanho exato do `rank`. Eles só podem ser chamados uma vez: de novo, devolvem `..._ALREADY_ALLOCATED`. O `alloc_data()` precisa do shape antes, porque é dele que sai `n`. Chamar `alloc_data()` de novo não aloca outra vez. O destrutor libera `data`, `shape` e `stride`.

O lançamento de um kernel é assíncrono: `mat_mul` e `hadamard_mul` voltam antes de o kernel terminar. Um erro *dentro* do kernel só aparece numa sincronização, que é o que o `device_to_host` faz antes de copiar. Para conferir sem copiar, use `DEVICE::synchronize()`. Os dois devolvem `TENSOR_EXEC_FAILED` nesse caso.

## Cópia proibida

O `TENSOR` é dono de `data`, `shape` e `stride`: o destrutor libera essa memória. Se ele pudesse ser copiado, as duas cópias liberariam a mesma memória no fim do escopo (*double free*). Por isso a cópia é erro de compilação:

```cpp
DEVICE::TENSOR b = a;     // erro: use of deleted function
void f(DEVICE::TENSOR t); // passar por valor também não compila
```

Passe o `TENSOR` por ponteiro ou referência. Para ver a mesma memória de outro jeito (por exemplo, transposta), converta para `TENSOR3` com `to_tensor3` e use `transpose`: o `TENSOR3` não é dono de nada, então copiá-lo é seguro.