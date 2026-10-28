// Software binary64 and expf. check-float.py checks binary64 results exactly
// and requires exp32 results to match host libm within 1 ULP.

fn clz32(value: u32) -> u32 {
  if (value == 0u) { return 32u; }
  var bits = value;
  var count = 0u;
  if ((bits & 0xffff0000u) == 0u) { count += 16u; bits <<= 16u; }
  if ((bits & 0xff000000u) == 0u) { count += 8u; bits <<= 8u; }
  if ((bits & 0xf0000000u) == 0u) { count += 4u; bits <<= 4u; }
  if ((bits & 0xc0000000u) == 0u) { count += 2u; bits <<= 2u; }
  if ((bits & 0x80000000u) == 0u) { count += 1u; }
  return count;
}

fn clz64(value: vec2<u32>) -> u32 {
  if (value.y != 0u) { return clz32(value.y); }
  return 32u + clz32(value.x);
}

fn shr64(value: vec2<u32>, amount: u32) -> vec2<u32> {
  if (amount == 0u) { return value; }
  if (amount >= 64u) { return vec2<u32>(0u, 0u); }
  if (amount >= 32u) { return vec2<u32>(value.y >> (amount - 32u), 0u); }
  return vec2<u32>((value.x >> amount) | (value.y << (32u - amount)), value.y >> amount);
}

fn shl64(value: vec2<u32>, amount: u32) -> vec2<u32> {
  if (amount == 0u) { return value; }
  if (amount >= 64u) { return vec2<u32>(0u, 0u); }
  if (amount >= 32u) { return vec2<u32>(0u, value.x << (amount - 32u)); }
  return vec2<u32>(value.x << amount, (value.y << amount) | (value.x >> (32u - amount)));
}

fn u64_or(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(left.x | right.x, left.y | right.y);
}

fn u64_xor(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(left.x ^ right.x, left.y ^ right.y);
}

fn u64_and(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(left.x & right.x, left.y & right.y);
}

fn u64_add(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  let low = left.x + right.x;
  let carry = select(0u, 1u, low < left.x);
  return vec2<u32>(low, left.y + right.y + carry);
}

fn u64_sub(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  let borrow = select(0u, 1u, left.x < right.x);
  return vec2<u32>(left.x - right.x, left.y - right.y - borrow);
}

fn u64_eq(left: vec2<u32>, right: vec2<u32>) -> bool {
  return left.x == right.x && left.y == right.y;
}

fn u64_lt(left: vec2<u32>, right: vec2<u32>) -> bool {
  return left.y < right.y || (left.y == right.y && left.x < right.x);
}

fn u64_nz(value: vec2<u32>) -> bool {
  return value.x != 0u || value.y != 0u;
}

fn bit64(index: u32) -> vec2<u32> {
  if (index >= 32u) { return vec2<u32>(0u, 1u << (index - 32u)); }
  return vec2<u32>(1u << index, 0u);
}

fn shr128(value: vec4<u32>, amount: u32) -> vec4<u32> {
  if (amount == 0u) { return value; }
  if (amount >= 128u) { return vec4<u32>(0u, 0u, 0u, 0u); }
  let words = amount >> 5u;
  var shifted = value;
  if (words == 1u) { shifted = vec4<u32>(value.y, value.z, value.w, 0u); }
  if (words == 2u) { shifted = vec4<u32>(value.z, value.w, 0u, 0u); }
  if (words == 3u) { shifted = vec4<u32>(value.w, 0u, 0u, 0u); }
  let bits = amount & 31u;
  if (bits == 0u) { return shifted; }
  let next = vec4<u32>(shifted.y, shifted.z, shifted.w, 0u);
  return vec4<u32>(
    (shifted.x >> bits) | (next.x << (32u - bits)),
    (shifted.y >> bits) | (next.y << (32u - bits)),
    (shifted.z >> bits) | (next.z << (32u - bits)),
    shifted.w >> bits
  );
}

fn shl128(value: vec4<u32>, amount: u32) -> vec4<u32> {
  if (amount == 0u) { return value; }
  if (amount >= 128u) { return vec4<u32>(0u, 0u, 0u, 0u); }
  let words = amount >> 5u;
  var shifted = value;
  if (words == 1u) { shifted = vec4<u32>(0u, value.x, value.y, value.z); }
  if (words == 2u) { shifted = vec4<u32>(0u, 0u, value.x, value.y); }
  if (words == 3u) { shifted = vec4<u32>(0u, 0u, 0u, value.x); }
  let bits = amount & 31u;
  if (bits == 0u) { return shifted; }
  let previous = vec4<u32>(0u, shifted.x, shifted.y, shifted.z);
  return vec4<u32>(
    shifted.x << bits,
    (shifted.y << bits) | (previous.y >> (32u - bits)),
    (shifted.z << bits) | (previous.z >> (32u - bits)),
    (shifted.w << bits) | (previous.w >> (32u - bits))
  );
}

fn add128(left: vec4<u32>, right: vec4<u32>) -> vec4<u32> {
  let x = left.x + right.x;
  let cx = select(0u, 1u, x < left.x);
  let y_sum = left.y + right.y;
  let y = y_sum + cx;
  let cy = select(0u, 1u, y_sum < left.y) + select(0u, 1u, y < cx);
  let z_sum = left.z + right.z;
  let z = z_sum + cy;
  let cz = select(0u, 1u, z_sum < left.z) + select(0u, 1u, z < cy);
  let w = left.w + right.w + cz;
  return vec4<u32>(x, y, z, w);
}

fn sub128(left: vec4<u32>, right: vec4<u32>) -> vec4<u32> {
  let bx = select(0u, 1u, left.x < right.x);
  let x = left.x - right.x;
  let y_base = left.y - right.y;
  let by = select(0u, 1u, left.y < right.y) + select(0u, 1u, y_base < bx);
  let y = y_base - bx;
  let z_base = left.z - right.z;
  let bz = select(0u, 1u, left.z < right.z) + select(0u, 1u, z_base < by);
  let z = z_base - by;
  let w = left.w - right.w - bz;
  return vec4<u32>(x, y, z, w);
}

fn nz128(value: vec4<u32>) -> bool {
  return value.x != 0u || value.y != 0u || value.z != 0u || value.w != 0u;
}

fn low64(value: vec4<u32>) -> vec2<u32> {
  return vec2<u32>(value.x, value.y);
}

fn from_low64(value: vec2<u32>) -> vec4<u32> {
  return vec4<u32>(value.x, value.y, 0u, 0u);
}

fn mul32_wide(left: u32, right: u32) -> vec2<u32> {
  let left_low = left & 65535u;
  let left_high = left >> 16u;
  let right_low = right & 65535u;
  let right_high = right >> 16u;
  let p0 = left_low * right_low;
  let p1 = left_low * right_high;
  let p2 = left_high * right_low;
  let p3 = left_high * right_high;
  let middle = (p0 >> 16u) + (p1 & 65535u) + (p2 & 65535u);
  return vec2<u32>((p0 & 65535u) | ((middle & 65535u) << 16u), p3 + (p1 >> 16u) + (p2 >> 16u) + (middle >> 16u));
}

fn mul64(left: vec2<u32>, right: vec2<u32>) -> vec4<u32> {
  let ll = mul32_wide(left.x, right.x);
  let lh = mul32_wide(left.x, right.y);
  let hl = mul32_wide(left.y, right.x);
  let hh = mul32_wide(left.y, right.y);
  var low = vec2<u32>(ll.x, 0u);
  var mid = u64_add(vec2<u32>(ll.y, 0u), u64_add(lh, hl));
  var high = hh;
  // mid is bits 32..96, so its high word carries into bits 64..
  high = u64_add(high, vec2<u32>(mid.y, 0u));
  return vec4<u32>(low.x, mid.x, high.x, high.y);
}

fn f32_to_f64(bits: u32) -> vec2<u32> {
  let sign = bits & 0x80000000u;
  let exponent = (bits >> 23u) & 255u;
  var fraction = bits & 0x7fffffu;
  let sign64 = vec2<u32>(0u, sign);
  if (exponent == 255u) {
    let payload = shl64(vec2<u32>(fraction, 0u), 29u);
    return u64_or(sign64, u64_or(vec2<u32>(0u, 0x7ff00000u), payload));
  }
  if (exponent == 0u && fraction == 0u) { return sign64; }
  var unbiased = i32(exponent) - 127;
  if (exponent == 0u) {
    unbiased = -126;
    loop {
      if ((fraction & 0x800000u) != 0u) { break; }
      fraction <<= 1u;
      unbiased -= 1;
    }
    fraction &= 0x7fffffu;
  }
  let field = u32(unbiased + 1023);
  let significand = shl64(vec2<u32>(fraction, 0u), 29u);
  return u64_or(sign64, u64_or(vec2<u32>(0u, field << 20u), significand));
}

fn f64_to_f32(bits: vec2<u32>) -> u32 {
  let sign = bits.y & 0x80000000u;
  let exponent = i32((bits.y >> 20u) & 2047u);
  let fraction = u64_and(bits, vec2<u32>(0xffffffffu, 0x000fffffu));
  if (exponent == 2047) {
    if (u64_nz(fraction)) { return sign | 0x7fc00000u | (shr64(fraction, 29u).x); }
    return sign | 0x7f800000u;
  }
  if (exponent == 0 && !u64_nz(fraction)) { return sign; }
  var unbiased = exponent - 1023;
  var significand = fraction;
  if (exponent != 0) { significand = u64_or(fraction, bit64(52u)); }
  if (exponent == 0 && u64_nz(significand)) {
    let shift = clz64(significand) - 11u;
    significand = shl64(significand, shift);
    unbiased -= i32(shift);
  }
  if (unbiased > 127) { return sign | 0x7f800000u; }
  if (unbiased < -150) { return sign; }
  var shift = 29u;
  if (unbiased < -126) {
    shift += u32(-126 - unbiased);
    unbiased = -127;
  }
  if (shift >= 64u) { return sign; }
  var kept = shr64(significand, shift);
  let rest = u64_and(significand, u64_sub(shl64(vec2<u32>(1u, 0u), shift), vec2<u32>(1u, 0u)));
  let mid = bit64(shift - 1u);
  let round_up = u64_lt(mid, rest) || (u64_eq(rest, mid) && (kept.x & 1u) == 1u);
  if (round_up) { kept = u64_add(kept, vec2<u32>(1u, 0u)); }
  if (unbiased == -127) { return sign | kept.x; }
  if (kept.x == 0x1000000u && kept.y == 0u) {
    kept = shr64(kept, 1u);
    unbiased += 1;
  }
  if (unbiased > 127) { return sign | 0x7f800000u; }
  return sign | (u32(unbiased + 127) << 23u) | (kept.x & 0x7fffffu);
}

fn pack64(sign: u32, unbiased_in: i32, significand: vec2<u32>) -> vec2<u32> {
  var unbiased = unbiased_in;
  var sig = significand;
  let sign_bit = vec2<u32>(0u, sign << 31u);
  if (unbiased > 1023) { return u64_or(sign_bit, vec2<u32>(0u, 0x7ff00000u)); }
  if (!u64_nz(sig) || unbiased < -1074) { return sign_bit; }
  if (unbiased < -1022) {
    let shift = u32(-1022 - unbiased);
    if (shift >= 64u) { return sign_bit; }
    let rest = u64_and(sig, u64_sub(shl64(vec2<u32>(1u, 0u), shift), vec2<u32>(1u, 0u)));
    let mid = bit64(shift - 1u);
    var kept = shr64(sig, shift);
    if (u64_lt(mid, rest) || (u64_eq(rest, mid) && (kept.x & 1u) == 1u)) { kept = u64_add(kept, vec2<u32>(1u, 0u)); }
    return u64_or(sign_bit, kept);
  }
  if (!u64_lt(sig, bit64(53u))) {
    sig = shr64(sig, 1u);
    unbiased += 1;
  }
  if (unbiased > 1023) { return u64_or(sign_bit, vec2<u32>(0u, 0x7ff00000u)); }
  let field = vec2<u32>(0u, u32(unbiased + 1023) << 20u);
  let fraction = u64_and(sig, vec2<u32>(0xffffffffu, 0x000fffffu));
  return u64_or(sign_bit, u64_or(field, fraction));
}

struct Parts {
  sign: u32,
  unbiased: i32,
  sig: vec2<u32>,
}

fn unpack64(bits: vec2<u32>) -> Parts {
  let sign = bits.y >> 31u;
  let exponent = i32((bits.y >> 20u) & 2047u);
  let fraction = u64_and(bits, vec2<u32>(0xffffffffu, 0x000fffffu));
  if (exponent == 0) {
    if (!u64_nz(fraction)) { return Parts(sign, -1022, vec2<u32>(0u, 0u)); }
    let shift = clz64(fraction) - 11u;
    return Parts(sign, -1022 - i32(shift), shl64(fraction, shift));
  }
  return Parts(sign, exponent - 1023, u64_or(fraction, bit64(52u)));
}

fn round_pack(sign: u32, unbiased_in: i32, wide: vec4<u32>, point_in: i32) -> vec2<u32> {
  var unbiased = unbiased_in;
  var point = point_in;
  var extra = 0;
  if (unbiased < -1022) {
    extra = -1022 - unbiased;
    unbiased = -1022;
  }
  var shift = point + extra;
  var value = wide;
  if (shift < 0) {
    value = shl128(value, u32(-shift));
    shift = 0;
  }
  if (shift >= 128) { return vec2<u32>(0u, sign << 31u); }
  var kept = low64(shr128(value, u32(shift)));
  let rest_wide = vec4<u32>(value.x, value.y, value.z, value.w);
  let shifted = shl128(shr128(value, u32(shift)), u32(shift));
  // rest = value - shifted, but only the low `shift` bits; compare via mask
  var rest = vec4<u32>(0u, 0u, 0u, 0u);
  if (shift > 0) {
    let mask = sub128(shl128(vec4<u32>(1u, 0u, 0u, 0u), u32(shift)), vec4<u32>(1u, 0u, 0u, 0u));
    rest = vec4<u32>(value.x & mask.x, value.y & mask.y, value.z & mask.z, value.w & mask.w);
  }
  let mid = shl128(vec4<u32>(1u, 0u, 0u, 0u), u32(shift - 1));
  let rest_gt = rest.w > mid.w || (rest.w == mid.w && (rest.z > mid.z || (rest.z == mid.z && (rest.y > mid.y || (rest.y == mid.y && rest.x > mid.x)))));
  let rest_eq = rest.x == mid.x && rest.y == mid.y && rest.z == mid.z && rest.w == mid.w;
  if (shift > 0 && (rest_gt || (rest_eq && (kept.x & 1u) == 1u))) { kept = u64_add(kept, vec2<u32>(1u, 0u)); }
  if (extra == 0) {
    if (!u64_lt(kept, bit64(53u))) {
      kept = shr64(kept, 1u);
      unbiased += 1;
    }
    return pack64(sign, unbiased, kept);
  }
  if (!u64_lt(kept, bit64(52u))) { return pack64(sign, -1022, bit64(52u)); }
  return u64_or(vec2<u32>(0u, sign << 31u), kept);
}

fn add_f64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let a_exp = (a.y >> 20u) & 2047u;
  let b_exp = (b.y >> 20u) & 2047u;
  let a_frac = u64_and(a, vec2<u32>(0xffffffffu, 0x000fffffu));
  let b_frac = u64_and(b, vec2<u32>(0xffffffffu, 0x000fffffu));
  if (a_exp == 2047u) {
    if (u64_nz(a_frac) || (b_exp == 2047u && (u64_nz(b_frac) || (a.y >> 31u) != (b.y >> 31u)))) { return vec2<u32>(0u, 0x7ff80000u); }
    return a;
  }
  if (b_exp == 2047u) {
    if (u64_nz(b_frac)) { return vec2<u32>(0u, 0x7ff80000u); }
    return b;
  }
  var ap = unpack64(a);
  var bp = unpack64(b);
  if (!u64_nz(ap.sig) && !u64_nz(bp.sig)) { return vec2<u32>(0u, (a.y & b.y) & 0x80000000u); }
  if (!u64_nz(ap.sig)) { return b; }
  if (!u64_nz(bp.sig)) { return a; }
  if (ap.unbiased < bp.unbiased || (ap.unbiased == bp.unbiased && u64_lt(ap.sig, bp.sig))) {
    let swap_s = ap.sign; ap.sign = bp.sign; bp.sign = swap_s;
    let swap_e = ap.unbiased; ap.unbiased = bp.unbiased; bp.unbiased = swap_e;
    let swap_g = ap.sig; ap.sig = bp.sig; bp.sig = swap_g;
  }
  let shift = ap.unbiased - bp.unbiased;
  var x = shl128(from_low64(ap.sig), 60u);
  var y = shl128(from_low64(bp.sig), 60u);
  if (shift >= 120) {
    y = vec4<u32>(select(0u, 1u, nz128(y)), 0u, 0u, 0u);
  } else if (shift > 0) {
    y = shr128(y, u32(shift));
  }
  var sum = x;
  if (ap.sign == bp.sign) { sum = add128(x, y); } else { sum = sub128(x, y); }
  if (!nz128(sum)) { return vec2<u32>(0u, 0u); }
  var unbiased = ap.unbiased;
  var lead = 0;
  if (sum.w != 0u) { lead = 127 - i32(clz32(sum.w)); }
  else if (sum.z != 0u) { lead = 95 - i32(clz32(sum.z)); }
  else if (sum.y != 0u) { lead = 63 - i32(clz32(sum.y)); }
  else { lead = 31 - i32(clz32(sum.x)); }
  let delta = lead - 112;
  if (delta < 0) { sum = shl128(sum, u32(-delta)); }
  unbiased += delta;
  let point = 60 + max(delta, 0);
  return round_pack(ap.sign, unbiased, sum, point);
}

fn mul_f64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let a_exp = (a.y >> 20u) & 2047u;
  let b_exp = (b.y >> 20u) & 2047u;
  let a_frac = u64_and(a, vec2<u32>(0xffffffffu, 0x000fffffu));
  let b_frac = u64_and(b, vec2<u32>(0xffffffffu, 0x000fffffu));
  let sign = (a.y ^ b.y) >> 31u;
  if (a_exp == 2047u || b_exp == 2047u) {
    let a_zero = a_exp == 0u && !u64_nz(a_frac);
    let b_zero = b_exp == 0u && !u64_nz(b_frac);
    if ((a_exp == 2047u && u64_nz(a_frac)) || (b_exp == 2047u && u64_nz(b_frac)) || (a_exp == 2047u && b_zero) || (b_exp == 2047u && a_zero)) {
      return vec2<u32>(0u, 0x7ff80000u);
    }
    return vec2<u32>(0u, (sign << 31u) | 0x7ff00000u);
  }
  let ap = unpack64(a);
  let bp = unpack64(b);
  if (!u64_nz(ap.sig) || !u64_nz(bp.sig)) { return vec2<u32>(0u, sign << 31u); }
  let product = mul64(ap.sig, bp.sig);
  var unbiased = ap.unbiased + bp.unbiased;
  var point = 52;
  let high = (product.w >> 9u) & 1u;
  if (high == 1u) { point = 53; unbiased += 1; }
  return round_pack(sign, unbiased, product, point);
}

fn exp_tab(index: u32) -> vec2<u32> {
  switch index {
    case 0u: { return vec2<u32>(0x00000000u, 0x3ff00000u); }
    case 1u: { return vec2<u32>(0xd3158574u, 0x3fefd9b0u); }
    case 2u: { return vec2<u32>(0x6cf9890fu, 0x3fefb558u); }
    case 3u: { return vec2<u32>(0xd0125b51u, 0x3fef9301u); }
    case 4u: { return vec2<u32>(0x3c7d517bu, 0x3fef72b8u); }
    case 5u: { return vec2<u32>(0x3168b9aau, 0x3fef5487u); }
    case 6u: { return vec2<u32>(0x6e756238u, 0x3fef387au); }
    case 7u: { return vec2<u32>(0xf51fdee1u, 0x3fef1e9du); }
    case 8u: { return vec2<u32>(0x0a31b715u, 0x3fef06feu); }
    case 9u: { return vec2<u32>(0x373aa9cbu, 0x3feef1a7u); }
    case 10u: { return vec2<u32>(0x4c123422u, 0x3feedea6u); }
    case 11u: { return vec2<u32>(0x6061892du, 0x3feece08u); }
    case 12u: { return vec2<u32>(0xd5362a27u, 0x3feebfdau); }
    case 13u: { return vec2<u32>(0x569d4f82u, 0x3feeb42bu); }
    case 14u: { return vec2<u32>(0xdd485429u, 0x3feeab07u); }
    case 15u: { return vec2<u32>(0xb03a5585u, 0x3feea47eu); }
    case 16u: { return vec2<u32>(0x667f3bcdu, 0x3feea09eu); }
    case 17u: { return vec2<u32>(0xe8ec5f74u, 0x3fee9f75u); }
    case 18u: { return vec2<u32>(0x73eb0187u, 0x3feea114u); }
    case 19u: { return vec2<u32>(0x994cce13u, 0x3feea589u); }
    case 20u: { return vec2<u32>(0x422aa0dbu, 0x3feeace5u); }
    case 21u: { return vec2<u32>(0xb0cdc5e5u, 0x3feeb737u); }
    case 22u: { return vec2<u32>(0x82a3f090u, 0x3feec491u); }
    case 23u: { return vec2<u32>(0xb23e255du, 0x3feed503u); }
    case 24u: { return vec2<u32>(0x995ad3adu, 0x3feee89fu); }
    case 25u: { return vec2<u32>(0xf2fb5e47u, 0x3feeff76u); }
    case 26u: { return vec2<u32>(0xdd85529cu, 0x3fef199bu); }
    case 27u: { return vec2<u32>(0xdcef9069u, 0x3fef3720u); }
    case 28u: { return vec2<u32>(0xdcfba487u, 0x3fef5818u); }
    case 29u: { return vec2<u32>(0x337b9b5fu, 0x3fef7c97u); }
    case 30u: { return vec2<u32>(0xa2a490dau, 0x3fefa4afu); }
    default: { return vec2<u32>(0x5b6e4540u, 0x3fefd076u); }
  }
}

fn exp32(value: f32) -> f32 {
  let bits = bitcast<u32>(value);
  let abstop = (bits >> 20u) & 2047u;
  if (abstop >= 0x42bu) { // top12(88) roughly; exact check below uses the float
    if (bits == 0xff800000u) { return 0.0; }
    if (abstop >= 0x7f8u) { return value + value; }
    if (value > 88.7228394) { return bitcast<f32>(0x7f800000u | (bits & 0x80000000u)); }
    if (value < -103.972084) { return 0.0; }
  }
  let poly0 = vec2<u32>(0x4b912394u, 0x3ebc6af8u);
  let poly1 = vec2<u32>(0x50fac4f3u, 0x3f2ebfceu);
  let poly2 = vec2<u32>(0xff0c52d6u, 0x3f962e42u);
  let inv = vec2<u32>(0x652b82feu, 0x40471547u);
  let shift = vec2<u32>(0x00000000u, 0x43380000u);
  let one = vec2<u32>(0x00000000u, 0x3ff00000u);
  let z = mul_f64(inv, f32_to_f64(bits));
  var kd = add_f64(z, shift);
  let index = kd;
  kd = add_f64(kd, u64_xor(shift, vec2<u32>(0u, 0x80000000u)));
  let reduced = add_f64(z, u64_xor(kd, vec2<u32>(0u, 0x80000000u)));
  var entry = exp_tab(index.x & 31u);
  entry = u64_add(entry, shl64(index, 47u));
  let z2 = add_f64(mul_f64(poly0, reduced), poly1);
  let square = mul_f64(reduced, reduced);
  var y = add_f64(mul_f64(poly2, reduced), one);
  y = add_f64(mul_f64(z2, square), y);
  y = mul_f64(y, entry);
  return bitcast<f32>(f64_to_f32(y));
}
