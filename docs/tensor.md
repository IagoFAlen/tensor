# O que é cada coisa no tensor

Este documento mostra o que cada campo de [`HOST::tensor`](../include/host/tensor.hpp) e [`DEVICE::tensor`](../include/device/tensor.hpp) representa, usando o shape `[2, 3, 4]` como exemplo. Para ver como os strides são calculados, leia [row-major.md](row-major.md).

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
| `data` | Alocado na RAM com `malloc` | Alocado na GPU com `cudaMalloc` |
| Valores iniciais | Não preenche `data` | Copia de `h_data` para a GPU |
| `shape` | Recebe o ponteiro no construtor | Recebe o ponteiro no construtor |
| `stride` | Calcula no construtor | Recebe o ponteiro no construtor |

```text
          RAM                                 GPU

 HOST::tensor                          
 ├─ data ───► [ valores ]                     
 ├─ shape ──► [ 2, 3, 4 ]                     
 └─ stride ─► [ 12, 4, 1 ]                    
                  ▲                           
                  │ ponteiros recebidos       
 DEVICE::tensor   │ no construtor             
 ├─ shape ────────┤                           
 ├─ stride ───────┘                           
 └─ data ─────────────────────────────► [ valores copiados ]
```

O desenho mostra o caso em que `DEVICE::tensor` recebe os ponteiros de `shape` e `stride` de um `HOST::tensor`, o que depende de quem chama o construtor. Em qualquer caso, a própria struct `DEVICE::tensor` e esses dois arrays ficam na RAM, e só `data` fica na GPU.

## Limitações atuais

- **`shape`:** as structs guardam o ponteiro recebido, sem copiar o array, então ele precisa continuar válido enquanto o tensor existir.
- **`stride` no host:** é alocado no construtor, mas o destrutor não o libera.
