package gonka

// RIPEMD-160 implementation for Cosmos address derivation.
// Reference: https://homes.esat.kuleuven.be/~bosMDselaers/ripemd160.html

import (
	"encoding/binary"
	"math/bits"
)

func ripemd160Sum(data []byte) [20]byte {
	// Initial hash values.
	h0 := uint32(0x67452301)
	h1 := uint32(0xEFCDAB89)
	h2 := uint32(0x98BADCFE)
	h3 := uint32(0x10325476)
	h4 := uint32(0xC3D2E1F0)

	// Pre-processing: pad message.
	msgLen := len(data)
	data = append(data, 0x80)
	for len(data)%64 != 56 {
		data = append(data, 0x00)
	}
	// Append original length in bits as 64-bit little-endian.
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(msgLen)*8)
	data = append(data, lenBuf[:]...)

	// Process each 512-bit (64-byte) block.
	for i := 0; i < len(data); i += 64 {
		var x [16]uint32
		for j := 0; j < 16; j++ {
			x[j] = binary.LittleEndian.Uint32(data[i+j*4:])
		}

		al, bl, cl, dl, el := h0, h1, h2, h3, h4
		ar, br, cr, dr, er := h0, h1, h2, h3, h4

		// 80 rounds, left and right.
		for j := 0; j < 80; j++ {
			var fl, fr uint32
			var kl, kr uint32

			switch {
			case j < 16:
				fl = bl ^ cl ^ dl
				fr = br ^ (cr &^ er) | (dr & er) // br ^ (cr AND (NOT er)) | (dr AND er)  → actually: br ^ (cr & dr) ... let me use correct formula
				kl = 0x00000000
				kr = 0x50A28BE6
			case j < 32:
				fl = (bl & cl) | (^bl & dl)
				fr = (cr & er) | (^cr & dr) // wrong, let me fix
				kl = 0x5A827999
				kr = 0x5C4DD124
			case j < 48:
				fl = (bl | ^cl) ^ dl
				fr = (cr | ^er) ^ dr
				kl = 0x6ED9EBA1
				kr = 0x6D703EF3
			case j < 64:
				fl = (bl & dl) | (cl & ^dl)
				fr = (cr & dr) | (er & ^dr) // wrong
				kl = 0x8F1BBCDC
				kr = 0x7A6D76E9
			default:
				fl = bl ^ (cl | ^dl)
				fr = br ^ (cr | ^dr) // wrong
				kl = 0xA953FD4E
				kr = 0x00000000
			}

			// Fix right-lane boolean functions (reverse order of left).
			switch {
			case j < 16:
				fr = br ^ (cr | ^dr)
			case j < 32:
				fr = (br & dr) | (cr & ^dr)
			case j < 48:
				fr = (br | ^cr) ^ dr
			case j < 64:
				fr = (br & cr) | (^br & dr)
			default:
				fr = br ^ cr ^ dr
			}

			sl := ripemd160rl[j]
			sr := ripemd160rr[j]

			tl := bits.RotateLeft32(al+fl+x[sl]+kl, int(ripemd160sl[j])) + el
			al, el, dl, cl, bl = el, dl, bits.RotateLeft32(cl, 10), bl, tl

			tr := bits.RotateLeft32(ar+fr+x[sr]+kr, int(ripemd160sr[j])) + er
			ar, er, dr, cr, br = er, dr, bits.RotateLeft32(cr, 10), br, tr
		}

		t := h1 + cl + dr
		h1 = h2 + dl + er
		h2 = h3 + el + ar
		h3 = h4 + al + br
		h4 = h0 + bl + cr
		h0 = t
	}

	var digest [20]byte
	binary.LittleEndian.PutUint32(digest[0:], h0)
	binary.LittleEndian.PutUint32(digest[4:], h1)
	binary.LittleEndian.PutUint32(digest[8:], h2)
	binary.LittleEndian.PutUint32(digest[12:], h3)
	binary.LittleEndian.PutUint32(digest[16:], h4)
	return digest
}

// Message word selection (left rounds).
var ripemd160rl = [80]uint32{
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
	7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
	3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
	1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
	4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13,
}

// Message word selection (right rounds).
var ripemd160rr = [80]uint32{
	5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
	6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
	15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
	8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
	12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11,
}

// Rotation amounts (left rounds).
var ripemd160sl = [80]uint32{
	11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
	7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
	11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
	11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
	9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6,
}

// Rotation amounts (right rounds).
var ripemd160sr = [80]uint32{
	8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
	9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
	9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
	15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
	8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11,
}
