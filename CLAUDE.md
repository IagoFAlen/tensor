# Tensor

Biblioteca própria de tensores em C/C++ para CPU (`HOST`) e GPU (`DEVICE`, com CUDA).

## Escopo

- O CuDNN **não é dependência** do projeto: serve apenas como referência de design. Não proponha chamadas ao CuDNN como solução; implemente aqui.
- Ao citar o CuDNN, deixe claro que é uma comparação e não restrinja o projeto aos limites dele (por exemplo, o mínimo de 3 dimensões do `cudnnSetTensorNdDescriptor`).

## Modelo do tensor

- Layout row-major: `shape` vai da dimensão mais lenta para a mais rápida, e `stride` é medido em elementos.
- `rank` é o tamanho de `shape`.
- Conceitos em [docs/tensor.md](docs/tensor.md) e [docs/row-major.md](docs/row-major.md).

## Ambiente

- O `git` está instalado só no WSL. A partir do Windows, use `wsl git ...`.
