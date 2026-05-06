
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout (constant_id =  0) const uint32_t WorkGroupSize = 128;
layout (constant_id =  1) const uint32_t Br = 1;
layout (constant_id =  2) const uint32_t Bc = 32;
layout (constant_id =  3) const uint32_t HSK = 32;
layout (constant_id =  4) const uint32_t HSV = 32;
layout (constant_id =  5) const uint32_t Clamp = 0;
layout (constant_id =  6) const uint32_t D_split = 16;
layout (constant_id =  7) const uint32_t row_split = 1;
layout (constant_id =  8) const uint32_t SubGroupSize = 32;
layout (constant_id =  9) const uint32_t SHMEM_STAGING = 0;
layout (constant_id = 10) const uint32_t Flags = 0;
layout (constant_id = 11) const uint32_t LIMIT_OCCUPANCY_SHMEM = 0;

const bool USE_MASK_OPT    = (Flags & 1) != 0;
const bool MASK_ENABLE     = (Flags & 2) != 0;
const bool LOGIT_SOFTCAP   = (Flags & 4) != 0;
const bool OLD_AMD_WINDOWS = (Flags & 8) != 0;

// Round up head sizes to a multiple of 16, for coopmat1/coopmat2 paths
const uint32_t HSK_pad = (HSK + 15) & ~15;
const uint32_t HSV_pad = (HSV + 15) & ~15;

const bool KV_bounds_check = Clamp != 0;

layout (push_constant) uniform parameter {
    uint32_t N;
    uint32_t KV;

    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;

    uint32_t neq2;
    uint32_t neq3;
    uint32_t nek2;
    uint32_t nek3;
    uint32_t nev2;
    uint32_t nev3;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nem3;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t nb21;
    uint32_t nb22;
    uint32_t nb23;

    float scale;
    float max_bias;
    float logit_softcap;

    uint32_t mask_n_head_log2;
    float m0;
    float m1;

    uint32_t gqa_ratio;
    uint32_t split_kv;
    uint32_t k_num;
} p;

#define SINK_ENABLE_BIT (1<<24)
#define N_LOG2_MASK 0xFFFF

layout (binding = 4) readonly buffer S {float data_s[];};

layout (binding = 5) writeonly buffer O {D_TYPE data_o[];};
layout (binding = 5) writeonly buffer OV4 {D_TYPEV4 data_ov4[];};

layout (binding = 6) readonly buffer MO {uint32_t data_mask_opt[];};

#define MASK_OPT_ALL_NEG_INF 1
#define MASK_OPT_ALL_ZERO 2

#define BINDING_IDX_K 0
#define BINDING_IDX_V 1
#if defined(DATA_A_F32)
layout (binding = 1) readonly buffer K_PACKED {vec4 k_data_packed[];} k_packed;
layout (binding = 2) readonly buffer V_PACKED {vec4 v_data_packed[];} v_packed;
#elif defined(A_TYPE_PACKED16)
layout (binding = 1) readonly buffer K_PACKED16 {A_TYPE_PACKED16 k_data_packed16[];} k_packed;
layout (binding = 2) readonly buffer V_PACKED16 {A_TYPE_PACKED16 v_data_packed16[];} v_packed;
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1
#endif

#if defined(DATA_A_F32)
#undef BLOCK_SIZE
#define BLOCK_SIZE 4
#define BLOCK_BYTE_SIZE 16

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    // iqs is currently always zero in the flash attention shaders
    if (binding_idx == BINDING_IDX_K) {
        return FLOAT_TYPEV4(k_packed.k_data_packed[a_offset + ib]);
    } else {
        return FLOAT_TYPEV4(v_packed.v_data_packed[a_offset + ib]);
    }
}
#endif

#if defined(DATA_A_Q4_0)
#define BLOCK_BYTE_SIZE 18
#elif defined(DATA_A_Q4_1)
#define BLOCK_BYTE_SIZE 20
#endif

#if defined(DATA_A_Q4_0) || defined(DATA_A_Q4_1)
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        uint vui_lo = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF);
#ifdef DATA_A_Q4_1
        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * nibbles + FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].m);
#else
        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * (nibbles - FLOAT_TYPE(8.0f));
#endif
    } else {
        uint vui_lo = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF);
#ifdef DATA_A_Q4_1
        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * nibbles + FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].m);
#else
        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * (nibbles - FLOAT_TYPE(8.0f));
#endif
    }
}
#endif

#if defined(DATA_A_Q5_0)
#define BLOCK_BYTE_SIZE 22
#elif defined(DATA_A_Q5_1)
#define BLOCK_BYTE_SIZE 24
#endif

#if defined(DATA_A_Q5_0) || defined(DATA_A_Q5_1)
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        uint vui_lo = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

#ifdef DATA_A_Q5_1
        uint qh = k_packed.k_data_packed16[a_offset + ib].qh;
#else
        uint qh = uint(k_packed.k_data_packed16[a_offset + ib].qh[0]) | (uint(k_packed.k_data_packed16[a_offset + ib].qh[1]) << 16);
#endif
        FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs) & 1, (qh >> (iqs + 1)) & 1, (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1) * FLOAT_TYPE(16.0f);

        FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF);
#ifdef DATA_A_Q5_1
        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * (nibbles + hb) + FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].m);
#else
        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * (nibbles + hb - FLOAT_TYPE(16.0f));
#endif
    } else {
        uint vui_lo = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

#ifdef DATA_A_Q5_1
        uint qh = v_packed.v_data_packed16[a_offset + ib].qh;
#else
        uint qh = uint(v_packed.v_data_packed16[a_offset + ib].qh[0]) | (uint(v_packed.v_data_packed16[a_offset + ib].qh[1]) << 16);
#endif
        FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs) & 1, (qh >> (iqs + 1)) & 1, (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1) * FLOAT_TYPE(16.0f);

        FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF);
#ifdef DATA_A_Q5_1
        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * (nibbles + hb) + FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].m);
#else
        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * (nibbles + hb - FLOAT_TYPE(16.0f));
#endif
    }
}
#endif


#if defined(DATA_A_IQ4_NL)
#define BLOCK_BYTE_SIZE 18

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        uint vui_lo = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * FLOAT_TYPEV4(
            kvalues_iq4nl[vui_lo & 0xF],
            kvalues_iq4nl[(vui_lo >> 8) & 0xF],
            kvalues_iq4nl[vui_hi & 0xF],
            kvalues_iq4nl[(vui_hi >> 8) & 0xF]);
    } else {
        uint vui_lo = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(v_packed.v_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * FLOAT_TYPEV4(
            kvalues_iq4nl[vui_lo & 0xF],
            kvalues_iq4nl[(vui_lo >> 8) & 0xF],
            kvalues_iq4nl[vui_hi & 0xF],
            kvalues_iq4nl[(vui_hi >> 8) & 0xF]);
    }
}
#endif
#if defined(DATA_A_Q8_0)
#define BLOCK_BYTE_SIZE 34
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        const i8vec2 v0 = unpack8(int32_t(k_packed.k_data_packed16[a_offset + ib].qs[iqs / 2])).xy; // vec4 used due to #12147
        const i8vec2 v1 = unpack8(int32_t(k_packed.k_data_packed16[a_offset + ib].qs[iqs / 2 + 1])).xy;

        return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d) * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);
    } else {
        const i8vec2 v0 = unpack8(int32_t(v_packed.v_data_packed16[a_offset + ib].qs[iqs / 2])).xy; // vec4 used due to #12147
        const i8vec2 v1 = unpack8(int32_t(v_packed.v_data_packed16[a_offset + ib].qs[iqs / 2 + 1])).xy;

        return FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d) * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);
    }
}
#endif

#if defined(DATA_A_TQ3_0)
#define BLOCK_BYTE_SIZE 14

// TQ3_0 codebook centroids
const FLOAT_TYPE tq3_centroids_fa[8] = FLOAT_TYPE[8](
    FLOAT_TYPE(-2.1519454), FLOAT_TYPE(-1.3439092), FLOAT_TYPE(-0.7560052), FLOAT_TYPE(-0.2450942),
    FLOAT_TYPE( 0.2450942), FLOAT_TYPE( 0.7560052), FLOAT_TYPE( 1.3439092), FLOAT_TYPE( 2.1519454)
);

#ifdef SEMI_QUANT_DISABLE
// A/B mode: classic dequantize4 path for TQ3_0 (same data format, standard compute)
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    uint group = iqs >> 3u;
    uint within = iqs & 7u;
    uint base = group * 3u;
    uint w0, w1;
    FLOAT_TYPE sc;
    if (binding_idx == BINDING_IDX_K) {
        w0 = uint(k_packed.k_data_packed16[a_offset + ib].qs[base / 2u]);
        w1 = uint(k_packed.k_data_packed16[a_offset + ib].qs[base / 2u + 1u]);
        sc = FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d);
    } else {
        w0 = uint(v_packed.v_data_packed16[a_offset + ib].qs[base / 2u]);
        w1 = uint(v_packed.v_data_packed16[a_offset + ib].qs[base / 2u + 1u]);
        sc = FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d);
    }
    uint b0, b1, b2;
    if ((base & 1u) == 0u) {
        b0 = w0 & 0xFFu; b1 = (w0 >> 8) & 0xFFu; b2 = w1 & 0xFFu;
    } else {
        b0 = (w0 >> 8) & 0xFFu; b1 = w1 & 0xFFu; b2 = (w1 >> 8) & 0xFFu;
    }
    uint bits24 = b0 | (b1 << 8) | (b2 << 16);
    uint shift = (within == 0u) ? 0u : 12u;
    return sc * FLOAT_TYPEV4(
        tq3_centroids_fa[(bits24 >> shift) & 7u],
        tq3_centroids_fa[(bits24 >> (shift + 3u)) & 7u],
        tq3_centroids_fa[(bits24 >> (shift + 6u)) & 7u],
        tq3_centroids_fa[(bits24 >> (shift + 9u)) & 7u]);
}
#else
// No intermediate FLOAT_TYPEV4 is constructed — each centroid lookup is immediately
// multiplied with the corresponding Q element and accumulated as a scalar.
ACC_TYPE semi_quant_qk_dot(uint ib, uint iqs, uint a_offset, FLOAT_TYPEV4 q) {
    uint group = iqs >> 3u;
    uint within = iqs & 7u;
    uint base = group * 3u;
    uint w0 = uint(k_packed.k_data_packed16[a_offset + ib].qs[base / 2u]);
    uint w1 = uint(k_packed.k_data_packed16[a_offset + ib].qs[base / 2u + 1u]);
    uint b0, b1, b2;
    if ((base & 1u) == 0u) {
        b0 = w0 & 0xFFu; b1 = (w0 >> 8) & 0xFFu; b2 = w1 & 0xFFu;
    } else {
        b0 = (w0 >> 8) & 0xFFu; b1 = w1 & 0xFFu; b2 = (w1 >> 8) & 0xFFu;
    }
    uint bits24 = b0 | (b1 << 8) | (b2 << 16);
    uint shift = (within == 0u) ? 0u : 12u;
    ACC_TYPE k_sc = ACC_TYPE(k_packed.k_data_packed16[a_offset + ib].d);
    // Fused index-extract → centroid-lookup → Q-multiply → accumulate
    ACC_TYPE acc = ACC_TYPE(0.0);
    acc += ACC_TYPE(q.x) * ACC_TYPE(tq3_centroids_fa[(bits24 >> shift) & 7u]);
    acc += ACC_TYPE(q.y) * ACC_TYPE(tq3_centroids_fa[(bits24 >> (shift + 3u)) & 7u]);
    acc += ACC_TYPE(q.z) * ACC_TYPE(tq3_centroids_fa[(bits24 >> (shift + 6u)) & 7u]);
    acc += ACC_TYPE(q.w) * ACC_TYPE(tq3_centroids_fa[(bits24 >> (shift + 9u)) & 7u]);
    return k_sc * acc;
}

// Semi-quantized PV accumulation: accumulates P*V into output directly from packed
// TQ3_0 data. Each centroid is looked up, scaled, multiplied by P, and accumulated
// per-element — no intermediate FLOAT_TYPEV4 V vector is constructed.
void semi_quant_pv_accum(uint ib, uint iqs, uint a_offset, FLOAT_TYPE p_val, inout FLOAT_TYPEV4 out_acc) {
    uint group = iqs >> 3u;
    uint within = iqs & 7u;
    uint base = group * 3u;
    uint w0 = uint(v_packed.v_data_packed16[a_offset + ib].qs[base / 2u]);
    uint w1 = uint(v_packed.v_data_packed16[a_offset + ib].qs[base / 2u + 1u]);
    uint b0, b1, b2;
    if ((base & 1u) == 0u) {
        b0 = w0 & 0xFFu; b1 = (w0 >> 8) & 0xFFu; b2 = w1 & 0xFFu;
    } else {
        b0 = (w0 >> 8) & 0xFFu; b1 = w1 & 0xFFu; b2 = (w1 >> 8) & 0xFFu;
    }
    uint bits24 = b0 | (b1 << 8) | (b2 << 16);
    uint shift = (within == 0u) ? 0u : 12u;
    FLOAT_TYPE sv = p_val * FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d);
    // Fused index-extract → centroid-lookup → scale*P multiply → accumulate
    out_acc.x += sv * tq3_centroids_fa[(bits24 >> shift) & 7u];
    out_acc.y += sv * tq3_centroids_fa[(bits24 >> (shift + 3u)) & 7u];
    out_acc.z += sv * tq3_centroids_fa[(bits24 >> (shift + 6u)) & 7u];
    out_acc.w += sv * tq3_centroids_fa[(bits24 >> (shift + 9u)) & 7u];
}
#endif // SEMI_QUANT_DISABLE
#endif

#if defined(DATA_A_TQ2_0)
#define BLOCK_BYTE_SIZE 10

#ifdef SEMI_QUANT_DISABLE
// A/B mode: classic dequantize4 path for TQ2_0 (same data format, standard compute)
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    FLOAT_TYPE scale;
    uint b;
    if (binding_idx == BINDING_IDX_K) {
        scale = FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d);
        uint w = uint(k_packed.k_data_packed16[a_offset + ib].qs[iqs >> 3u]);
        if ((iqs & 4u) == 0u) { b = w & 0xFFu; }
        else                  { b = (w >> 8u) & 0xFFu; }
    } else {
        scale = FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d);
        uint w = uint(v_packed.v_data_packed16[a_offset + ib].qs[iqs >> 3u]);
        if ((iqs & 4u) == 0u) { b = w & 0xFFu; }
        else                  { b = (w >> 8u) & 0xFFu; }
    }
    return (scale * FLOAT_TYPE(0.333333343)) * FLOAT_TYPEV4(
        FLOAT_TYPE(int( b        & 3u) * 2 - 3),
        FLOAT_TYPE(int((b >> 2u) & 3u) * 2 - 3),
        FLOAT_TYPE(int((b >> 4u) & 3u) * 2 - 3),
        FLOAT_TYPE(int((b >> 6u) & 3u) * 2 - 3));
}
#else
// Semi-quantized QK dot for TQ2_0: computes dot(Q, K) directly from packed 2-bit indices.
// Uses algebraic centroids {-1, -1/3, +1/3, +1} via (idx*2-3) * scale * 0.333.
ACC_TYPE semi_quant_qk_dot(uint ib, uint iqs, uint a_offset, FLOAT_TYPEV4 q) {
    FLOAT_TYPE scale = FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d);
    uint w = uint(k_packed.k_data_packed16[a_offset + ib].qs[iqs >> 3u]);
    uint b;
    if ((iqs & 4u) == 0u) { b = w & 0xFFu; }
    else                  { b = (w >> 8u) & 0xFFu; }
    ACC_TYPE sv = ACC_TYPE(scale) * ACC_TYPE(0.333333343);
    ACC_TYPE acc = ACC_TYPE(0.0);
    acc += ACC_TYPE(q.x) * ACC_TYPE(int( b        & 3u) * 2 - 3);
    acc += ACC_TYPE(q.y) * ACC_TYPE(int((b >> 2u) & 3u) * 2 - 3);
    acc += ACC_TYPE(q.z) * ACC_TYPE(int((b >> 4u) & 3u) * 2 - 3);
    acc += ACC_TYPE(q.w) * ACC_TYPE(int((b >> 6u) & 3u) * 2 - 3);
    return sv * acc;
}

// Semi-quantized PV accumulation from packed TQ2_0 data. Each index is extracted,
// converted to centroid, scaled by P, and accumulated per-element — no FLOAT_TYPEV4 V vector.
void semi_quant_pv_accum(uint ib, uint iqs, uint a_offset, FLOAT_TYPE p_val, inout FLOAT_TYPEV4 out_acc) {
    FLOAT_TYPE scale = FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d);
    uint w = uint(v_packed.v_data_packed16[a_offset + ib].qs[iqs >> 3u]);
    uint b;
    if ((iqs & 4u) == 0u) { b = w & 0xFFu; }
    else                  { b = (w >> 8u) & 0xFFu; }
    FLOAT_TYPE sv = p_val * scale * FLOAT_TYPE(0.333333343);
    // Fused index-extract → centroid → P*scale multiply → accumulate
    out_acc.x += sv * FLOAT_TYPE(int(b & 3u) * 2 - 3);
    out_acc.y += sv * FLOAT_TYPE(int((b >> 2u) & 3u) * 2 - 3);
    out_acc.z += sv * FLOAT_TYPE(int((b >> 4u) & 3u) * 2 - 3);
    out_acc.w += sv * FLOAT_TYPE(int((b >> 6u) & 3u) * 2 - 3);
}
#endif // SEMI_QUANT_DISABLE
#endif

// Mixed KV: K=TQ3_0 (3-bit, 8 centroids) + V=TQ2_0 (2-bit, 4-level algebraic)
#if defined(MIXED_KV_TQ3K_TQ2V)
#define K_BLOCK_BYTE_SIZE 14
#define V_BLOCK_BYTE_SIZE 10

layout (binding = 1) readonly buffer K_PACKED16 {block_tq3_0_packed16 k_data_packed16[];} k_packed;
layout (binding = 2) readonly buffer V_PACKED16 {block_tq2_0_packed16 v_data_packed16[];} v_packed;

const FLOAT_TYPE tq3_centroids_fa[8] = FLOAT_TYPE[8](
    FLOAT_TYPE(-2.1519454), FLOAT_TYPE(-1.3439092), FLOAT_TYPE(-0.7560052), FLOAT_TYPE(-0.2450942),
    FLOAT_TYPE( 0.2450942), FLOAT_TYPE( 0.7560052), FLOAT_TYPE( 1.3439092), FLOAT_TYPE( 2.1519454)
);

#ifdef SEMI_QUANT_DISABLE
// A/B mode: classic dequantize4 path for mixed KV (K=TQ3_0, V=TQ2_0)
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        // TQ3_0 K dequantization
        uint group = iqs >> 3u;
        uint within = iqs & 7u;
        uint base = group * 3u;
        uint w0 = uint(k_packed.k_data_packed16[a_offset + ib].qs[base / 2u]);
        uint w1 = uint(k_packed.k_data_packed16[a_offset + ib].qs[base / 2u + 1u]);
        uint b0, b1, b2;
        if ((base & 1u) == 0u) {
            b0 = w0 & 0xFFu; b1 = (w0 >> 8) & 0xFFu; b2 = w1 & 0xFFu;
        } else {
            b0 = (w0 >> 8) & 0xFFu; b1 = w1 & 0xFFu; b2 = (w1 >> 8) & 0xFFu;
        }
        uint bits24 = b0 | (b1 << 8) | (b2 << 16);
        uint shift = (within == 0u) ? 0u : 12u;
        FLOAT_TYPE sc = FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d);
        return sc * FLOAT_TYPEV4(
            tq3_centroids_fa[(bits24 >> shift) & 7u],
            tq3_centroids_fa[(bits24 >> (shift + 3u)) & 7u],
            tq3_centroids_fa[(bits24 >> (shift + 6u)) & 7u],
            tq3_centroids_fa[(bits24 >> (shift + 9u)) & 7u]);
    } else {
        // TQ2_0 V dequantization
        FLOAT_TYPE scale = FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d);
        uint w = uint(v_packed.v_data_packed16[a_offset + ib].qs[iqs >> 3u]);
        uint b;
        if ((iqs & 4u) == 0u) { b = w & 0xFFu; }
        else                  { b = (w >> 8u) & 0xFFu; }
        return (scale * FLOAT_TYPE(0.333333343)) * FLOAT_TYPEV4(
            FLOAT_TYPE(int( b        & 3u) * 2 - 3),
            FLOAT_TYPE(int((b >> 2u) & 3u) * 2 - 3),
            FLOAT_TYPE(int((b >> 4u) & 3u) * 2 - 3),
            FLOAT_TYPE(int((b >> 6u) & 3u) * 2 - 3));
    }
}
#else
// Semi-quantized QK dot for mixed KV: computes dot(Q, K) directly from packed TQ3_0 K bits.
// Fused index-extract → centroid-lookup → Q-multiply → accumulate. No FLOAT_TYPEV4 K vector.
ACC_TYPE semi_quant_qk_dot(uint ib, uint iqs, uint a_offset, FLOAT_TYPEV4 q) {
    uint group = iqs >> 3u;
    uint within = iqs & 7u;
    uint base = group * 3u;
    uint w0 = uint(k_packed.k_data_packed16[a_offset + ib].qs[base / 2u]);
    uint w1 = uint(k_packed.k_data_packed16[a_offset + ib].qs[base / 2u + 1u]);
    uint b0, b1, b2;
    if ((base & 1u) == 0u) {
        b0 = w0 & 0xFFu; b1 = (w0 >> 8) & 0xFFu; b2 = w1 & 0xFFu;
    } else {
        b0 = (w0 >> 8) & 0xFFu; b1 = w1 & 0xFFu; b2 = (w1 >> 8) & 0xFFu;
    }
    uint bits24 = b0 | (b1 << 8) | (b2 << 16);
    uint shift = (within == 0u) ? 0u : 12u;
    ACC_TYPE k_sc = ACC_TYPE(k_packed.k_data_packed16[a_offset + ib].d);
    ACC_TYPE acc = ACC_TYPE(0.0);
    acc += ACC_TYPE(q.x) * ACC_TYPE(tq3_centroids_fa[(bits24 >> shift) & 7u]);
    acc += ACC_TYPE(q.y) * ACC_TYPE(tq3_centroids_fa[(bits24 >> (shift + 3u)) & 7u]);
    acc += ACC_TYPE(q.z) * ACC_TYPE(tq3_centroids_fa[(bits24 >> (shift + 6u)) & 7u]);
    acc += ACC_TYPE(q.w) * ACC_TYPE(tq3_centroids_fa[(bits24 >> (shift + 9u)) & 7u]);
    return k_sc * acc;
}

// Semi-quantized PV accumulation for mixed KV: TQ2_0 V path.
// Fused index-extract → centroid → P*scale multiply → accumulate. No FLOAT_TYPEV4 V vector.
void semi_quant_pv_accum(uint ib, uint iqs, uint a_offset, FLOAT_TYPE p_val, inout FLOAT_TYPEV4 out_acc) {
    FLOAT_TYPE scale = FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d);
    uint w = uint(v_packed.v_data_packed16[a_offset + ib].qs[iqs >> 3u]);
    uint b;
    if ((iqs & 4u) == 0u) { b = w & 0xFFu; }
    else                  { b = (w >> 8u) & 0xFFu; }
    FLOAT_TYPE sv = p_val * scale * FLOAT_TYPE(0.333333343);
    out_acc.x += sv * FLOAT_TYPE(int(b & 3u) * 2 - 3);
    out_acc.y += sv * FLOAT_TYPE(int((b >> 2u) & 3u) * 2 - 3);
    out_acc.z += sv * FLOAT_TYPE(int((b >> 4u) & 3u) * 2 - 3);
    out_acc.w += sv * FLOAT_TYPE(int((b >> 6u) & 3u) * 2 - 3);
}
#endif // SEMI_QUANT_DISABLE
#endif

// Mixed KV: K=TQ3_0 (3-bit, 8 centroids) + V=TQ3_2 (2-bit, TQ2_0-format storage)
// TQ3_2 v2 — DATA-DRIVEN per-quad error compensation (Option B from spec):
//   * No tuned constants. The correction strength is derived from the actual per-quad bit
//     population, which is a direct estimator of per-quad quantization-error magnitude.
//   * K-side: TQ3_0 8-level Gaussian Lloyd-Max centroids. Per-quad, count K values at extreme
//     bins (idx 0 or 7, ±2.15σ centroid). These bins absorb the entire tail of the Gaussian
//     (bin 0 ≈ (-∞, -1.85σ], bin 7 ≈ [+1.85σ, +∞)) so each extreme value carries the largest
//     conditional reconstruction error (E[|x-c|² | x ∈ tail] ≈ 1.6× E[|x-c|²|x ∈ inner]).
//     Compensation: boost the per-quad QK contribution by (1 + 0.02 * n_extreme), which
//     scales monotonically with the actual error in this quad. n_extreme is computed from
//     the bits, not pre-tuned.
//   * V-side: TQ2_0 4-level uniform centroids. Per-quad, count outer levels (|r|=3). Outer
//     bins span (-∞,-2/3·amax] and [+2/3·amax,+∞), again absorbing the full tail with the
//     largest conditional error. Apply local 3-tap smoothing whose strength is the outer
//     fraction: alpha = 0.03125 * n_outer ∈ [0, 0.125]. Quads with no clipped values get
//     identity passthrough (alpha=0); quads with full saturation get full smoothing.
// Storage, VRAM, and bandwidth IDENTICAL to TQ3_1 (K=TQ3_0 + V=TQ2_0). Compute: ~6 ALU/quad.
#if defined(MIXED_KV_TQ3K_TQ3_2V)
#define K_BLOCK_BYTE_SIZE 14
#define V_BLOCK_BYTE_SIZE 10
// Maximum-correction scalars. NOT tuned per model — these are the upper bounds derived from
// the relative tail-vs-bulk MSE ratios of the respective quantizers (TQ3_0 8-level Gaussian
// LM tail/bulk error ratio ≈ 1.6 → max boost ≈ 8%; TQ2_0 4-level outer-bin variance ≈ 4×
// inner → max alpha ≈ 0.125 to halve outer-bin variance).
#define TQ3_2_QK_PER_EXTREME 0.02
#define TQ3_2_V_PER_OUTER    0.03125

layout (binding = 1) readonly buffer K_PACKED16 {block_tq3_0_packed16 k_data_packed16[];} k_packed;
layout (binding = 2) readonly buffer V_PACKED16 {block_tq3_2_packed16 v_data_packed16[];} v_packed;

const FLOAT_TYPE tq3_centroids_fa[8] = FLOAT_TYPE[8](
    FLOAT_TYPE(-2.1519454), FLOAT_TYPE(-1.3439092), FLOAT_TYPE(-0.7560052), FLOAT_TYPE(-0.2450942),
    FLOAT_TYPE( 0.2450942), FLOAT_TYPE( 0.7560052), FLOAT_TYPE( 1.3439092), FLOAT_TYPE( 2.1519454)
);

#ifdef SEMI_QUANT_DISABLE
// A/B mode: classic dequantize4 path for mixed KV (K=TQ3_0, V=TQ3_2)
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        // TQ3_0 K dequantization
        uint group = iqs >> 3u;
        uint within = iqs & 7u;
        uint base = group * 3u;
        uint w0 = uint(k_packed.k_data_packed16[a_offset + ib].qs[base / 2u]);
        uint w1 = uint(k_packed.k_data_packed16[a_offset + ib].qs[base / 2u + 1u]);
        uint b0, b1, b2;
        if ((base & 1u) == 0u) {
            b0 = w0 & 0xFFu; b1 = (w0 >> 8) & 0xFFu; b2 = w1 & 0xFFu;
        } else {
            b0 = (w0 >> 8) & 0xFFu; b1 = w1 & 0xFFu; b2 = (w1 >> 8) & 0xFFu;
        }
        uint bits24 = b0 | (b1 << 8) | (b2 << 16);
        uint shift = (within == 0u) ? 0u : 12u;
        FLOAT_TYPE sc = FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d);
        return sc * FLOAT_TYPEV4(
            tq3_centroids_fa[(bits24 >> shift) & 7u],
            tq3_centroids_fa[(bits24 >> (shift + 3u)) & 7u],
            tq3_centroids_fa[(bits24 >> (shift + 6u)) & 7u],
            tq3_centroids_fa[(bits24 >> (shift + 9u)) & 7u]);
    } else {
        // TQ3_2 V dequantization (TQ2_0 format)
        FLOAT_TYPE scale = FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d);
        uint w = uint(v_packed.v_data_packed16[a_offset + ib].qs[iqs >> 3u]);
        uint b;
        if ((iqs & 4u) == 0u) { b = w & 0xFFu; }
        else                  { b = (w >> 8u) & 0xFFu; }
        return (scale * FLOAT_TYPE(0.333333343)) * FLOAT_TYPEV4(
            FLOAT_TYPE(int( b        & 3u) * 2 - 3),
            FLOAT_TYPE(int((b >> 2u) & 3u) * 2 - 3),
            FLOAT_TYPE(int((b >> 4u) & 3u) * 2 - 3),
            FLOAT_TYPE(int((b >> 6u) & 3u) * 2 - 3));
    }
}
#else
// Semi-quantized QK dot for mixed KV: TQ3_0 K decode
ACC_TYPE semi_quant_qk_dot(uint ib, uint iqs, uint a_offset, FLOAT_TYPEV4 q) {
    uint group = iqs >> 3u;
    uint within = iqs & 7u;
    uint base = group * 3u;
    uint w0 = uint(k_packed.k_data_packed16[a_offset + ib].qs[base / 2u]);
    uint w1 = uint(k_packed.k_data_packed16[a_offset + ib].qs[base / 2u + 1u]);
    uint b0, b1, b2;
    if ((base & 1u) == 0u) {
        b0 = w0 & 0xFFu; b1 = (w0 >> 8) & 0xFFu; b2 = w1 & 0xFFu;
    } else {
        b0 = (w0 >> 8) & 0xFFu; b1 = w1 & 0xFFu; b2 = (w1 >> 8) & 0xFFu;
    }
    uint bits24 = b0 | (b1 << 8) | (b2 << 16);
    uint shift = (within == 0u) ? 0u : 12u;
    ACC_TYPE k_sc = ACC_TYPE(k_packed.k_data_packed16[a_offset + ib].d);
    uint i0 = (bits24 >> shift)            & 7u;
    uint i1 = (bits24 >> (shift + 3u))     & 7u;
    uint i2 = (bits24 >> (shift + 6u))     & 7u;
    uint i3 = (bits24 >> (shift + 9u))     & 7u;
    ACC_TYPE acc = ACC_TYPE(0.0);
    acc += ACC_TYPE(q.x) * ACC_TYPE(tq3_centroids_fa[i0]);
    acc += ACC_TYPE(q.y) * ACC_TYPE(tq3_centroids_fa[i1]);
    acc += ACC_TYPE(q.z) * ACC_TYPE(tq3_centroids_fa[i2]);
    acc += ACC_TYPE(q.w) * ACC_TYPE(tq3_centroids_fa[i3]);
    // Data-driven QK correction: count K values at extreme bins (idx 0 or 7).
    // Each extreme contributes the largest reconstruction error → boost proportionally.
    uint n_extreme = uint(i0 == 0u || i0 == 7u)
                   + uint(i1 == 0u || i1 == 7u)
                   + uint(i2 == 0u || i2 == 7u)
                   + uint(i3 == 0u || i3 == 7u);
    ACC_TYPE qk_scale = ACC_TYPE(1.0) + ACC_TYPE(TQ3_2_QK_PER_EXTREME) * ACC_TYPE(n_extreme);
    return k_sc * acc * qk_scale;
}

// Semi-quantized PV accumulation for mixed KV (K=TQ3_0, V=TQ3_2)
// Per-quad data-driven smoothing: alpha = TQ3_2_V_PER_OUTER * count(|r|=3 in this quad).
void semi_quant_pv_accum(uint ib, uint iqs, uint a_offset, FLOAT_TYPE p_val, inout FLOAT_TYPEV4 out_acc) {
    FLOAT_TYPE scale = FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d);
    uint w = uint(v_packed.v_data_packed16[a_offset + ib].qs[iqs >> 3u]);
    uint b;
    if ((iqs & 4u) == 0u) { b = w & 0xFFu; }
    else                  { b = (w >> 8u) & 0xFFu; }
    FLOAT_TYPE sv = p_val * scale * FLOAT_TYPE(0.333333343);
    int r0 = int( b        & 3u) * 2 - 3;
    int r1 = int((b >> 2u) & 3u) * 2 - 3;
    int r2 = int((b >> 4u) & 3u) * 2 - 3;
    int r3 = int((b >> 6u) & 3u) * 2 - 3;
    FLOAT_TYPE c0 = FLOAT_TYPE(r0);
    FLOAT_TYPE c1 = FLOAT_TYPE(r1);
    FLOAT_TYPE c2 = FLOAT_TYPE(r2);
    FLOAT_TYPE c3 = FLOAT_TYPE(r3);
    // Data-driven alpha: proportional to outer-level count in THIS quad.
    uint n_outer = uint(abs(r0) == 3) + uint(abs(r1) == 3)
                 + uint(abs(r2) == 3) + uint(abs(r3) == 3);
    FLOAT_TYPE a = FLOAT_TYPE(TQ3_2_V_PER_OUTER) * FLOAT_TYPE(n_outer);
    FLOAT_TYPE one_minus_2a = FLOAT_TYPE(1.0) - FLOAT_TYPE(2.0) * a;
    FLOAT_TYPE c1f = one_minus_2a * c1 + a * (c0 + c2);
    FLOAT_TYPE c2f = one_minus_2a * c2 + a * (c1 + c3);
    out_acc.x += sv * c0;
    out_acc.y += sv * c1f;
    out_acc.z += sv * c2f;
    out_acc.w += sv * c3;
}
#endif // SEMI_QUANT_DISABLE
#endif

// TQ3_2 standalone: both K and V use TQ2_0-format blocks (for -ctk tq3_2 -ctv tq3_2)
// v2 — DATA-DRIVEN per-quad corrections (parity with the mixed-KV path):
//   * K-side is 2-bit algebraic here (no 8-level codebook), so the
//     extreme-bin proxy is |r|==3 (matches V-side). Correction:
//     qk_scale = 1 + TQ3_2_QK_PER_OUTER * n_K_outer.
//   * V-side: alpha = TQ3_2_V_PER_OUTER * n_V_outer (same as mixed path).
#if defined(DATA_A_TQ3_2)
#define BLOCK_BYTE_SIZE 10
#define TQ3_2_QK_PER_OUTER 0.03125
#define TQ3_2_V_PER_OUTER  0.03125

#ifdef SEMI_QUANT_DISABLE
// A/B mode: classic dequantize4 path for TQ3_2 (identical to TQ2_0)
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    FLOAT_TYPE scale;
    uint b;
    if (binding_idx == BINDING_IDX_K) {
        scale = FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d);
        uint w = uint(k_packed.k_data_packed16[a_offset + ib].qs[iqs >> 3u]);
        if ((iqs & 4u) == 0u) { b = w & 0xFFu; }
        else                  { b = (w >> 8u) & 0xFFu; }
    } else {
        scale = FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d);
        uint w = uint(v_packed.v_data_packed16[a_offset + ib].qs[iqs >> 3u]);
        if ((iqs & 4u) == 0u) { b = w & 0xFFu; }
        else                  { b = (w >> 8u) & 0xFFu; }
    }
    return (scale * FLOAT_TYPE(0.333333343)) * FLOAT_TYPEV4(
        FLOAT_TYPE(int( b        & 3u) * 2 - 3),
        FLOAT_TYPE(int((b >> 2u) & 3u) * 2 - 3),
        FLOAT_TYPE(int((b >> 4u) & 3u) * 2 - 3),
        FLOAT_TYPE(int((b >> 6u) & 3u) * 2 - 3));
}
#else
// Semi-quantized QK dot for TQ3_2 standalone (K uses TQ2_0-format, 4 algebraic centroids).
// v2: data-driven correction from outer-level count on K side.
ACC_TYPE semi_quant_qk_dot(uint ib, uint iqs, uint a_offset, FLOAT_TYPEV4 q) {
    FLOAT_TYPE scale = FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d);
    uint w = uint(k_packed.k_data_packed16[a_offset + ib].qs[iqs >> 3u]);
    uint b;
    if ((iqs & 4u) == 0u) { b = w & 0xFFu; }
    else                  { b = (w >> 8u) & 0xFFu; }
    int r0 = int( b        & 3u) * 2 - 3;
    int r1 = int((b >> 2u) & 3u) * 2 - 3;
    int r2 = int((b >> 4u) & 3u) * 2 - 3;
    int r3 = int((b >> 6u) & 3u) * 2 - 3;
    ACC_TYPE sv = ACC_TYPE(scale) * ACC_TYPE(0.333333343);
    ACC_TYPE acc = ACC_TYPE(0.0);
    acc += ACC_TYPE(q.x) * ACC_TYPE(r0);
    acc += ACC_TYPE(q.y) * ACC_TYPE(r1);
    acc += ACC_TYPE(q.z) * ACC_TYPE(r2);
    acc += ACC_TYPE(q.w) * ACC_TYPE(r3);
    uint n_outer = uint(abs(r0) == 3) + uint(abs(r1) == 3)
                 + uint(abs(r2) == 3) + uint(abs(r3) == 3);
    ACC_TYPE qk_scale = ACC_TYPE(1.0) + ACC_TYPE(TQ3_2_QK_PER_OUTER) * ACC_TYPE(n_outer);
    return sv * acc * qk_scale;
}

// Semi-quantized PV accumulation for TQ3_2 standalone.
// v2: data-driven per-quad alpha proportional to outer-level count.
void semi_quant_pv_accum(uint ib, uint iqs, uint a_offset, FLOAT_TYPE p_val, inout FLOAT_TYPEV4 out_acc) {
    FLOAT_TYPE scale = FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d);
    uint w = uint(v_packed.v_data_packed16[a_offset + ib].qs[iqs >> 3u]);
    uint b;
    if ((iqs & 4u) == 0u) { b = w & 0xFFu; }
    else                  { b = (w >> 8u) & 0xFFu; }
    FLOAT_TYPE sv = p_val * scale * FLOAT_TYPE(0.333333343);
    int r0 = int( b        & 3u) * 2 - 3;
    int r1 = int((b >> 2u) & 3u) * 2 - 3;
    int r2 = int((b >> 4u) & 3u) * 2 - 3;
    int r3 = int((b >> 6u) & 3u) * 2 - 3;
    FLOAT_TYPE c0 = FLOAT_TYPE(r0);
    FLOAT_TYPE c1 = FLOAT_TYPE(r1);
    FLOAT_TYPE c2 = FLOAT_TYPE(r2);
    FLOAT_TYPE c3 = FLOAT_TYPE(r3);
    uint n_outer = uint(abs(r0) == 3) + uint(abs(r1) == 3)
                 + uint(abs(r2) == 3) + uint(abs(r3) == 3);
    FLOAT_TYPE a = FLOAT_TYPE(TQ3_2_V_PER_OUTER) * FLOAT_TYPE(n_outer);
    FLOAT_TYPE one_minus_2a = FLOAT_TYPE(1.0) - FLOAT_TYPE(2.0) * a;
    FLOAT_TYPE c1f = one_minus_2a * c1 + a * (c0 + c2);
    FLOAT_TYPE c2f = one_minus_2a * c2 + a * (c1 + c3);
    out_acc.x += sv * c0;
    out_acc.y += sv * c1f;
    out_acc.z += sv * c2f;
    out_acc.w += sv * c3;
}
#endif // SEMI_QUANT_DISABLE
#endif

// =====================================================================
// TQ3_3 — DISABLED in the public KV cache selector (see common/arg.cpp).
// =====================================================================
// Status: the dedicated MIXED_KV_TQ3K_TQ3_3V shader variant is ~15% slower
// than the functionally-identical MIXED_KV_TQ3K_TQ2V path, even when the
// GLSL bodies are made byte-for-byte equivalent (diagnostic baseline below,
// verified in bench_tq3_3_baseline.json). The throughput penalty originates
// below the GLSL layer (SPIR-V compilation / driver pipeline artifact) and
// cannot be closed from this file.
//
// TQ3_3 as currently defined in ggml-common.h / ggml.c is a typedef alias
// of TQ3_2 (same block struct, same CPU quantize/dequantize). Therefore a
// real, distinct TQ3_3 requires one of:
//   * A new storage format (e.g. 18 B / 64-element shared-scale V block →
//     ~10% bandwidth reduction) — needs new CPU block struct, quantize,
//     dequantize, vec_dot, plus an FA kernel that tolerates mismatched K/V
//     element-per-block counts.
//   * A runtime bandwidth-reduction mechanism whose benefit exceeds the
//     15% variant-dispatch penalty on this model (the attention-aware
//     pruning experiment did not achieve this at 2 K context).
// Both are multi-session engineering tasks and are tracked in the project
// report as the real TQ3_3 roadmap.
//
// Until that lands, the block below remains compiled but dormant: it
// exists only so the existing Vulkan plumbing (pipeline tables, dispatch,
// shader-gen entries) keeps building cleanly. Users cannot reach it from
// `-ctv tq3_3` because the type is commented out in kv_cache_types.
// =====================================================================

#if defined(MIXED_KV_TQ3K_TQ3_3V)
// DIAGNOSTIC BASELINE: byte-identical to MIXED_KV_TQ3K_TQ2V. If TQ3_3 matches
// TQ3_1 throughput with this body, the prior 15% gap was a SPIR-V compile
// cache / variant-naming artifact, and architectural mechanism can now be
// layered on top from a known-good baseline.
#define K_BLOCK_BYTE_SIZE 14
#define V_BLOCK_BYTE_SIZE 10

layout (binding = 1) readonly buffer K_PACKED16_T33 {block_tq3_0_packed16 k_data_packed16[];} k_packed;
layout (binding = 2) readonly buffer V_PACKED16_T33 {block_tq2_0_packed16 v_data_packed16[];} v_packed;

const FLOAT_TYPE tq3_centroids_fa_33[8] = FLOAT_TYPE[8](
    FLOAT_TYPE(-2.1519454), FLOAT_TYPE(-1.3439092), FLOAT_TYPE(-0.7560052), FLOAT_TYPE(-0.2450942),
    FLOAT_TYPE( 0.2450942), FLOAT_TYPE( 0.7560052), FLOAT_TYPE( 1.3439092), FLOAT_TYPE( 2.1519454)
);

#ifdef SEMI_QUANT_DISABLE
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        uint group = iqs >> 3u;
        uint within = iqs & 7u;
        uint base = group * 3u;
        uint w0 = uint(k_packed.k_data_packed16[a_offset + ib].qs[base / 2u]);
        uint w1 = uint(k_packed.k_data_packed16[a_offset + ib].qs[base / 2u + 1u]);
        uint b0, b1, b2;
        if ((base & 1u) == 0u) { b0 = w0 & 0xFFu; b1 = (w0 >> 8) & 0xFFu; b2 = w1 & 0xFFu; }
        else                   { b0 = (w0 >> 8) & 0xFFu; b1 = w1 & 0xFFu; b2 = (w1 >> 8) & 0xFFu; }
        uint bits24 = b0 | (b1 << 8) | (b2 << 16);
        uint shift = (within == 0u) ? 0u : 12u;
        FLOAT_TYPE sc = FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d);
        return sc * FLOAT_TYPEV4(
            tq3_centroids_fa_33[(bits24 >> shift) & 7u],
            tq3_centroids_fa_33[(bits24 >> (shift + 3u)) & 7u],
            tq3_centroids_fa_33[(bits24 >> (shift + 6u)) & 7u],
            tq3_centroids_fa_33[(bits24 >> (shift + 9u)) & 7u]);
    } else {
        FLOAT_TYPE scale = FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d);
        uint w = uint(v_packed.v_data_packed16[a_offset + ib].qs[iqs >> 3u]);
        uint b;
        if ((iqs & 4u) == 0u) { b = w & 0xFFu; }
        else                  { b = (w >> 8u) & 0xFFu; }
        return (scale * FLOAT_TYPE(0.333333343)) * FLOAT_TYPEV4(
            FLOAT_TYPE(int( b        & 3u) * 2 - 3),
            FLOAT_TYPE(int((b >> 2u) & 3u) * 2 - 3),
            FLOAT_TYPE(int((b >> 4u) & 3u) * 2 - 3),
            FLOAT_TYPE(int((b >> 6u) & 3u) * 2 - 3));
    }
}
#else
ACC_TYPE semi_quant_qk_dot(uint ib, uint iqs, uint a_offset, FLOAT_TYPEV4 q) {
    uint group = iqs >> 3u;
    uint within = iqs & 7u;
    uint base = group * 3u;
    uint w0 = uint(k_packed.k_data_packed16[a_offset + ib].qs[base / 2u]);
    uint w1 = uint(k_packed.k_data_packed16[a_offset + ib].qs[base / 2u + 1u]);
    uint b0, b1, b2;
    if ((base & 1u) == 0u) {
        b0 = w0 & 0xFFu; b1 = (w0 >> 8) & 0xFFu; b2 = w1 & 0xFFu;
    } else {
        b0 = (w0 >> 8) & 0xFFu; b1 = w1 & 0xFFu; b2 = (w1 >> 8) & 0xFFu;
    }
    uint bits24 = b0 | (b1 << 8) | (b2 << 16);
    uint shift = (within == 0u) ? 0u : 12u;
    ACC_TYPE k_sc = ACC_TYPE(k_packed.k_data_packed16[a_offset + ib].d);
    ACC_TYPE acc = ACC_TYPE(0.0);
    acc += ACC_TYPE(q.x) * ACC_TYPE(tq3_centroids_fa_33[(bits24 >> shift) & 7u]);
    acc += ACC_TYPE(q.y) * ACC_TYPE(tq3_centroids_fa_33[(bits24 >> (shift + 3u)) & 7u]);
    acc += ACC_TYPE(q.z) * ACC_TYPE(tq3_centroids_fa_33[(bits24 >> (shift + 6u)) & 7u]);
    acc += ACC_TYPE(q.w) * ACC_TYPE(tq3_centroids_fa_33[(bits24 >> (shift + 9u)) & 7u]);
    return k_sc * acc;
}

void semi_quant_pv_accum(uint ib, uint iqs, uint a_offset, FLOAT_TYPE p_val, inout FLOAT_TYPEV4 out_acc) {
    FLOAT_TYPE scale = FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d);
    uint w = uint(v_packed.v_data_packed16[a_offset + ib].qs[iqs >> 3u]);
    uint b;
    if ((iqs & 4u) == 0u) { b = w & 0xFFu; }
    else                  { b = (w >> 8u) & 0xFFu; }
    FLOAT_TYPE sv = p_val * scale * FLOAT_TYPE(0.333333343);
    out_acc.x += sv * FLOAT_TYPE(int(b & 3u) * 2 - 3);
    out_acc.y += sv * FLOAT_TYPE(int((b >> 2u) & 3u) * 2 - 3);
    out_acc.z += sv * FLOAT_TYPE(int((b >> 4u) & 3u) * 2 - 3);
    out_acc.w += sv * FLOAT_TYPE(int((b >> 6u) & 3u) * 2 - 3);
}
#endif // SEMI_QUANT_DISABLE
#endif // MIXED_KV_TQ3K_TQ3_3V

// TQ3_3 standalone (both K and V = TQ3_3 format; rarely used since K benefits
// from TQ3_0's 3-bit codebook). Same Lloyd-Max V decode; K uses algebraic.
#if defined(DATA_A_TQ3_3)
#define BLOCK_BYTE_SIZE 10
#define TQ3_3_QK_SCALE 1.03

#ifdef SEMI_QUANT_DISABLE
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    FLOAT_TYPE scale;
    uint b;
    if (binding_idx == BINDING_IDX_K) {
        scale = FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d);
        uint w = uint(k_packed.k_data_packed16[a_offset + ib].qs[iqs >> 3u]);
        if ((iqs & 4u) == 0u) { b = w & 0xFFu; } else { b = (w >> 8u) & 0xFFu; }
    } else {
        scale = FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d);
        uint w = uint(v_packed.v_data_packed16[a_offset + ib].qs[iqs >> 3u]);
        if ((iqs & 4u) == 0u) { b = w & 0xFFu; } else { b = (w >> 8u) & 0xFFu; }
    }
    return (scale * FLOAT_TYPE(0.333333343)) * FLOAT_TYPEV4(
        FLOAT_TYPE(int( b        & 3u) * 2 - 3),
        FLOAT_TYPE(int((b >> 2u) & 3u) * 2 - 3),
        FLOAT_TYPE(int((b >> 4u) & 3u) * 2 - 3),
        FLOAT_TYPE(int((b >> 6u) & 3u) * 2 - 3));
}
#else
ACC_TYPE semi_quant_qk_dot(uint ib, uint iqs, uint a_offset, FLOAT_TYPEV4 q) {
    FLOAT_TYPE scale = FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d);
    uint w = uint(k_packed.k_data_packed16[a_offset + ib].qs[iqs >> 3u]);
    uint b;
    if ((iqs & 4u) == 0u) { b = w & 0xFFu; }
    else                  { b = (w >> 8u) & 0xFFu; }
    ACC_TYPE sv = ACC_TYPE(scale) * ACC_TYPE(0.333333343);
    ACC_TYPE acc = ACC_TYPE(0.0);
    acc += ACC_TYPE(q.x) * ACC_TYPE(int( b        & 3u) * 2 - 3);
    acc += ACC_TYPE(q.y) * ACC_TYPE(int((b >> 2u) & 3u) * 2 - 3);
    acc += ACC_TYPE(q.z) * ACC_TYPE(int((b >> 4u) & 3u) * 2 - 3);
    acc += ACC_TYPE(q.w) * ACC_TYPE(int((b >> 6u) & 3u) * 2 - 3);
    return sv * acc * ACC_TYPE(TQ3_3_QK_SCALE);
}

void semi_quant_pv_accum(uint ib, uint iqs, uint a_offset, FLOAT_TYPE p_val, inout FLOAT_TYPEV4 out_acc) {
    FLOAT_TYPE scale = FLOAT_TYPE(v_packed.v_data_packed16[a_offset + ib].d);
    uint w = uint(v_packed.v_data_packed16[a_offset + ib].qs[iqs >> 3u]);
    uint b;
    if ((iqs & 4u) == 0u) { b = w & 0xFFu; }
    else                  { b = (w >> 8u) & 0xFFu; }
    FLOAT_TYPE sv = p_val * scale * FLOAT_TYPE(0.333333343);
    int r0 = int( b        & 3u) * 2 - 3;
    int r1 = int((b >> 2u) & 3u) * 2 - 3;
    int r2 = int((b >> 4u) & 3u) * 2 - 3;
    int r3 = int((b >> 6u) & 3u) * 2 - 3;
    FLOAT_TYPE m0 = (abs(r0) == 3) ? FLOAT_TYPE(1.510) : FLOAT_TYPE(1.358);
    FLOAT_TYPE m1 = (abs(r1) == 3) ? FLOAT_TYPE(1.510) : FLOAT_TYPE(1.358);
    FLOAT_TYPE m2 = (abs(r2) == 3) ? FLOAT_TYPE(1.510) : FLOAT_TYPE(1.358);
    FLOAT_TYPE m3 = (abs(r3) == 3) ? FLOAT_TYPE(1.510) : FLOAT_TYPE(1.358);
    out_acc.x += sv * FLOAT_TYPE(r0) * m0;
    out_acc.y += sv * FLOAT_TYPE(r1) * m1;
    out_acc.z += sv * FLOAT_TYPE(r2) * m2;
    out_acc.w += sv * FLOAT_TYPE(r3) * m3;
}
#endif // SEMI_QUANT_DISABLE
#endif // DATA_A_TQ3_3

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))


// Store column zero. This is used to save per-row m and L values for split_k.
ACC_TYPE perElemOpStoreCol0(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    if (r < N && c == 0) {
        uint32_t offset = iq2 + r;
        data_o[o_offset + offset] = D_TYPE(elem);
    }
    return elem;
}

// Load the slope matrix, indexed by Q's dimension 2.
ACC_TYPE perElemOpComputeSlope(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    uint32_t n_head_log2 = p.mask_n_head_log2 & N_LOG2_MASK;

    const ACC_TYPE base = ACC_TYPE(h < n_head_log2 ? p.m0 : p.m1);
    const int      exph = int(h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1);

    return ACC_TYPE(pow(base, ACC_TYPE(exph)));
}

// Load the sink value, indexed by Q's dimension 2.
ACC_TYPE perElemOpGetSink(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    return ACC_TYPE(data_s[h]);
}

uint32_t i, N, KV, split_k_index, Tr, start_j, end_j,
         gqa_iq1, iq2, iq3, rk2, rk3, rv2, rv3, ik2, ik3, iv2, iv3,
         q_stride, k_stride, v_stride, m_stride;

void init_indices()
{
    N = p.N;
    KV = p.KV;

    if (p.k_num > 1) {
        if (p.gqa_ratio > 1) {
            i = 0;
            // batch and split_k share gl_WorkGroupID.x
            gqa_iq1 = gl_WorkGroupID.x / p.k_num;
            split_k_index = gl_WorkGroupID.x % p.k_num;
        } else {
            gqa_iq1 = 0;
            split_k_index = gl_WorkGroupID.x % p.k_num;
            i = gl_WorkGroupID.x / p.k_num;
        }
    } else if (p.gqa_ratio > 1) {
        i = 0;
        gqa_iq1 = gl_WorkGroupID.x;
        split_k_index = 0;
    } else {
        i = gl_WorkGroupID.x;
        gqa_iq1 = 0;
        split_k_index = 0;
    }

    Tr = CEIL_DIV(N, Br);

    start_j = split_k_index * p.split_kv / Bc;
    end_j = CEIL_DIV(min(KV, (split_k_index + 1) * p.split_kv), Bc);

    // When not using grouped query attention, all rows share the same iq2, equal to gl_WorkGroupID.y.
    // When using grouped query attention, each workgroup does gqa_ratio consecutive values of iq2.
    iq2 = gl_WorkGroupID.y * p.gqa_ratio;
    iq3 = gl_WorkGroupID.z;

    // broadcast factors
    rk2 = p.neq2/p.nek2;
    rk3 = p.neq3/p.nek3;

    rv2 = p.neq2/p.nev2;
    rv3 = p.neq3/p.nev3;

    // k indices
    ik3 = iq3 / rk3;
    ik2 = iq2 / rk2;

    // v indices
    iv3 = iq3 / rv3;
    iv2 = iq2 / rv2;

    // nb?1 are already divided by the type size and are in units of elements.
    // When using grouped query attention, Q is indexed by iq2, so the stride
    // should be nb02 (which is in bytes).
    q_stride = p.gqa_ratio > 1 ? (p.nb02 / 4) : p.nb01;
    k_stride = p.nb11;
    v_stride = p.nb21;
    // When using grouped query attention, all rows use the same mask (stride 0).
    // "p.gqa_ratio >> 16" is just a roundabout way of writing zero
    // that prevents the compiler from folding the "&" through the select
    // and breaking the alignment detection.
    m_stride = (p.gqa_ratio > 1) ? (p.gqa_ratio >> 16) : KV;
}

// Bias applied to softmax to stay in fp16 range.
// Based on ggml-cuda issue https://github.com/ggml-org/llama.cpp/issues/18606
const float FATTN_KQ_MAX_OFFSET = 3.0f*0.6931f;

// Store the output when doing grouped query attention.
// Rows index by Q's dimension 2, and the first N rows are valid.
void gqaStore(const in uint32_t r, const in uint32_t c, const in FLOAT_TYPEV4 elems, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    uint32_t offset = (iq2 + r) * HSV / 4 + c;
    data_ov4[o_offset + offset] = D_TYPEV4(elems);
}
