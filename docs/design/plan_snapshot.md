# MixFusion Token-Tile Plan Snapshot

This snapshot freezes the attention contract for the `mixfusion_both` and
`token_tile_both` variants.

```text
Q/K/V projection linears
    -> complete QK^T score matrix
    -> scale and mask
    -> softmax over the complete K/V range
    -> AV
    -> output projection linears
    -> residual and following transformer block
```

Token tiles are an execution plan for projection linears. They gather rows into
a packed tile, execute Base and LoRA through the same descriptor, and scatter
the result back to the original request rows. They do not split the attention
problem: the attention layer still receives the complete projected `Q`, `K`,
and `V` tensors and performs one softmax over the complete key/value range.

The implementation records this contract in the `attention_dependency`
telemetry event. A token-tile run is invalid if its attention event reports a
tile-local softmax, a tile-local K/V range, or a changed attention mask.
