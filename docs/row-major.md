# Row-major no `HOST::tensor`

Este documento mostra como o construtor de [`HOST::tensor`](../include/h_tensor.hpp) organiza um tensor na memória, usando o shape `[2, 3, 4]` como exemplo.

## A ideia

A memória é uma sequência linear, mas um tensor tem várias dimensões. Row-major é a regra que liga as duas coisas: **a última dimensão é a que varia mais rápido**, e a primeira, a que varia mais devagar.

É a convenção do comentário em `h_tensor.hpp`: as dimensões vão "do mais lento ao mais rápido".

```text
shape = [2, 3, 4]
         │  │  └─ dimensão 2: 4 elementos, a mais rápida
         │  └──── dimensão 1: 3 elementos
         └─────── dimensão 0: 2 elementos, a mais lenta
```

O tensor tem `2 × 3 × 4 = 24` elementos.

## Como os strides são calculados

O stride de uma dimensão é quantos elementos `float` é preciso avançar na memória para somar 1 ao índice dela.

O construtor preenche `stride` de trás para frente. A última posição vale 1, e cada uma das outras é o tamanho e o stride da dimensão seguinte multiplicados:

```cpp
size_t stride_count = rank - 1;

stride = (size_t*) malloc(rank * sizeof(size_t));
stride[stride_count] = 1;

for(size_t i = 0; i < rank; i++){
    n *= shape[i];

    if (stride_count > 0) {
        stride[stride_count - 1] = shape[stride_count] * stride[stride_count];
        stride_count--;
    }
}
```

Para o shape `[2, 3, 4]`:

| Passo | `stride_count` | Cálculo | Resultado |
|---|---|---|---|
| Antes do loop | 2 | Valor inicial | `stride[2] = 1` |
| `i = 0` | 2 | `shape[2] * stride[2] = 4 * 1` | `stride[1] = 4` |
| `i = 1` | 1 | `shape[1] * stride[1] = 3 * 4` | `stride[0] = 12` |
| `i = 2` | 0 | Nenhum | Sem alteração |

O resultado é `stride = [12, 4, 1]`:

- Somar 1 à dimensão 2 avança **1** elemento.
- Somar 1 à dimensão 1 pula um bloco de 4 elementos, avançando **4**.
- Somar 1 à dimensão 0 pula 3 blocos de 4 elementos, avançando **12**.

## De índice para posição

O projeto ainda não tem uma função de acesso, mas os strides levam a esta fórmula:

```text
posição = i * stride[0] + j * stride[1] + k * stride[2]
```

Para `[2, 3, 4]`, ela fica:

```text
posição = i * 12 + j * 4 + k
```

Por exemplo, o elemento `(1, 2, 3)` está na posição `1 * 12 + 2 * 4 + 3 = 23`, a última do tensor.

## O tensor na memória

Cada célula abaixo mostra a posição do elemento em `data`:

```text
i = 0                          i = 1

        k=0  k=1  k=2  k=3             k=0  k=1  k=2  k=3
j = 0     0    1    2    3     j = 0    12   13   14   15
j = 1     4    5    6    7     j = 1    16   17   18   19
j = 2     8    9   10   11     j = 2    20   21   22   23
```

Em `data`, os elementos ficam em sequência, sem separação:

```text
posição:  0  1  2  3 | 4  5  6  7 | 8  9 10 11 || 12 13 14 15 | 16 17 18 19 | 20 21 22 23
j:        ---- 0 --- | --- 1 ---- | ---- 2 --- || ---- 0 ---- | ---- 1 ---- | ---- 2 ----
i:        ---------------- 0 ----------------- || ----------------- 1 -----------------
```

As barras `|` e `||` só marcam os limites de `j` e `i` e não existem na memória.

Percorrer o tensor com a última dimensão no laço mais interno acessa as posições em ordem, de 0 a 23:

```cpp
for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 3; j++)
        for (size_t k = 0; k < 4; k++)
            // posição = i * 12 + j * 4 + k
```

## Limitação atual do construtor

`n` não é inicializado antes de `n *= shape[i]`. Assim, os strides são calculados corretamente, mas o valor de `n` e o tamanho alocado para `data` ficam indefinidos. Para o shape `[2, 3, 4]`, o valor correto de `n` seria 24.
