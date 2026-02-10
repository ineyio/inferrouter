package gonka

import "fmt"

const bech32Charset = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"

// bech32Encode encodes data (5-bit groups) with the given human-readable part.
func bech32Encode(hrp string, data []byte) (string, error) {
	values := append(bech32HRPExpand(hrp), data...)
	checksum := bech32Checksum(hrp, data)
	combined := append(data, checksum...)

	_ = values // used only conceptually for checksum

	result := make([]byte, 0, len(hrp)+1+len(combined))
	result = append(result, []byte(hrp)...)
	result = append(result, '1')
	for _, v := range combined {
		if int(v) >= len(bech32Charset) {
			return "", fmt.Errorf("gonka: bech32: invalid data byte %d", v)
		}
		result = append(result, bech32Charset[v])
	}
	return string(result), nil
}

func bech32HRPExpand(hrp string) []byte {
	expand := make([]byte, 0, len(hrp)*2+1)
	for _, c := range hrp {
		expand = append(expand, byte(c>>5))
	}
	expand = append(expand, 0)
	for _, c := range hrp {
		expand = append(expand, byte(c&31))
	}
	return expand
}

func bech32Polymod(values []byte) uint32 {
	gen := [5]uint32{0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3}
	chk := uint32(1)
	for _, v := range values {
		b := chk >> 25
		chk = (chk&0x1ffffff)<<5 ^ uint32(v)
		for i := 0; i < 5; i++ {
			if (b>>uint(i))&1 == 1 {
				chk ^= gen[i]
			}
		}
	}
	return chk
}

func bech32Checksum(hrp string, data []byte) []byte {
	values := append(bech32HRPExpand(hrp), data...)
	values = append(values, 0, 0, 0, 0, 0, 0)
	polymod := bech32Polymod(values) ^ 1
	checksum := make([]byte, 6)
	for i := 0; i < 6; i++ {
		checksum[i] = byte((polymod >> uint(5*(5-i))) & 31)
	}
	return checksum
}

// convertBits converts data between bit widths.
func convertBits(data []byte, fromBits, toBits uint, pad bool) ([]byte, error) {
	acc := uint32(0)
	bits := uint(0)
	maxv := uint32((1 << toBits) - 1)
	var result []byte

	for _, b := range data {
		if uint32(b)>>fromBits != 0 {
			return nil, fmt.Errorf("gonka: bech32: invalid data byte %d", b)
		}
		acc = (acc << fromBits) | uint32(b)
		bits += fromBits
		for bits >= toBits {
			bits -= toBits
			result = append(result, byte((acc>>bits)&maxv))
		}
	}

	if pad {
		if bits > 0 {
			result = append(result, byte((acc<<(toBits-bits))&maxv))
		}
	} else if bits >= fromBits {
		return nil, fmt.Errorf("gonka: bech32: excess padding")
	} else if (acc<<(toBits-bits))&maxv != 0 {
		return nil, fmt.Errorf("gonka: bech32: non-zero padding")
	}

	return result, nil
}
