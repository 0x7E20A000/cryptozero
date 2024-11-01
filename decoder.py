#!/usr/bin/env python3
import sys
import base64
import binascii
import zlib
import codecs
import urllib.parse
import json
import binascii
import re

from base64 import urlsafe_b64decode

def try_base64(s):
    try:
        padding = '=' * (-len(s) % 4)
        decoded = base64.b64decode(s + padding, validate=True)
        return decoded
    except Exception:
        return None

def try_base64_url(s):
    try:
        padding = '=' * (-len(s) % 4)
        decoded = urlsafe_b64decode(s + padding)
        return decoded
    except Exception:
        return None

def try_base32(s):
    try:
        padding = '=' * (-len(s) % 8)
        decoded = base64.b32decode(s + padding, casefold=True)
        return decoded
    except Exception:
        return None

def try_base16(s):
    try:
        decoded = base64.b16decode(s, casefold=True)
        return decoded
    except Exception:
        return None

def try_hex(s):
    try:
        decoded = bytes.fromhex(s)
        return decoded
    except ValueError:
        return None

def try_url_decode(s):
    try:
        decoded = urllib.parse.unquote(s)
        return decoded.encode('utf-8')
    except Exception:
        return None

def try_rot13(s):
    try:
        decoded = codecs.decode(s, 'rot_13')
        return decoded.encode('utf-8')
    except Exception:
        return None

def try_reverse(s):
    try:
        decoded = s[::-1]
        return decoded.encode('utf-8')
    except Exception:
        return None

def try_zlib(s):
    try:
        if isinstance(s, str):
            s = s.encode('utf-8')
        decompressed = zlib.decompress(s)
        return decompressed
    except zlib.error:
        return None

def try_gzip(s):
    import gzip
    import io
    try:
        if isinstance(s, str):
            s = s.encode('utf-8')
        with gzip.GzipFile(fileobj=io.BytesIO(s)) as f:
            decompressed = f.read()
            return decompressed
    except OSError:
        return None

def try_json(s):
    try:
        obj = json.loads(s)
        pretty = json.dumps(obj, indent=4)
        return pretty.encode('utf-8')
    except Exception:
        return None

def try_jwt(s):
    try:
        parts = s.split('.')
        if len(parts) != 3:
            return None
        header, payload, signature = parts
        decoded_header = base64url_decode(header).decode('utf-8', errors='replace')
        decoded_payload = base64url_decode(payload).decode('utf-8', errors='replace')
        return f"Header:\n{decoded_header}\n\nPayload:\n{decoded_payload}\n\nSignature:\n{signature}".encode('utf-8')
    except Exception:
        return None

def base64url_decode(input_str):
    padding = '=' * (-len(input_str) % 4)
    return base64.urlsafe_b64decode(input_str + padding)

def try_morse(s):
    MORSE_CODE_DICT = {
        'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
        'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
        'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
        'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
        'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
        'Z': '--..',
        '0': '-----', '1': '.----', '2': '..---', '3': '...--',
        '4': '....-', '5': '.....', '6': '-....', '7': '--...',
        '8': '---..', '9': '----.',
        '.': '.-.-.-', ',': '--..--', '?': '..--..', "'": '.----.',
        '!': '-.-.--', '/': '-..-.', '(': '-.--.', ')': '-.--.-',
        '&': '.-...', ':': '---...', ';': '-.-.-.', '=': '-...-',
        '+': '.-.-.', '-': '-....-', '_': '..--.-', '"': '.-..-.',
        '$': '...-..-', '@': '.--.-.', ' ': '/'
    }
    try:
        reverse_morse = {v: k for k, v in MORSE_CODE_DICT.items()}
        words = s.split('/')
        decoded = []
        for word in words:
            letters = word.split(' ')
            decoded_word = ''.join([reverse_morse.get(letter, '') for letter in letters])
            decoded.append(decoded_word)
        return ' '.join(decoded).encode('utf-8')
    except Exception:
        return None

def try_caesar(s, shift=13):
    try:
        decoded = ''.join([chr((ord(char) - 65 + shift) % 26 + 65) if char.isupper() else
                           chr((ord(char) - 97 + shift) % 26 + 97) if char.islower() else char
                           for char in s])
        return decoded.encode('utf-8')
    except Exception:
        return None

def try_xor(s, key=0xFF):
    try:
        if isinstance(s, str):
            s = s.encode('utf-8')
        decoded = bytes([b ^ key for b in s])
        return decoded
    except Exception:
        return None

def try_bin(s):
    try:
        decoded = int(s, 2).to_bytes((int(s, 2).bit_length() + 7) // 8, 'big')
        return decoded
    except Exception:
        return None

def try_oct(s):
    try:
        decoded = int(s, 8).to_bytes((int(s, 8).bit_length() + 7) // 8, 'big')
        return decoded
    except Exception:
        return None

def try_punycode(s):
    try:
        decoded = s.encode('ascii').decode('punycode')
        return decoded.encode('utf-8')
    except Exception:
        return None

def try_html_entities(s):
    try:
        decoded = html.unescape(s)
        return decoded.encode('utf-8')
    except Exception:
        return None

def try_brotli(s):
    try:
        import brotli
        if isinstance(s, str):
            s = s.encode('utf-8')
        decompressed = brotli.decompress(s)
        return decompressed
    except:
        return None

def try_lzma(s):
    try:
        import lzma
        if isinstance(s, str):
            s = s.encode('utf-8')
        decompressed = lzma.decompress(s)
        return decompressed
    except:
        return None

def try_all_decodings(s):
    results = {}

    # Base64
    res = try_base64(s)
    if res:
        results['Base64'] = res

    # Base64 URL-safe
    res = try_base64_url(s)
    if res:
        results['Base64 URL-safe'] = res

    # Base32
    res = try_base32(s)
    if res:
        results['Base32'] = res

    # Base16
    res = try_base16(s)
    if res:
        results['Base16'] = res

    # Hex
    res = try_hex(s)
    if res:
        results['Hex'] = res

    # URL Decode
    res = try_url_decode(s)
    if res:
        results['URL Decode'] = res

    # ROT13
    res = try_rot13(s)
    if res:
        results['ROT13'] = res

    # Reverse
    res = try_reverse(s)
    if res:
        results['Reverse'] = res

    # zlib Decompress
    res = try_zlib(s)
    if res:
        results['zlib Decompress'] = res

    # gzip Decompress
    res = try_gzip(s)
    if res:
        results['gzip Decompress'] = res

    # JSON Format
    res = try_json(s)
    if res:
        results['JSON'] = res

    # JWT Decode
    res = try_jwt(s)
    if res:
        results['JWT Decode'] = res

    # Morse Code
    res = try_morse(s)
    if res:
        results['Morse Code'] = res

    # Caesar Cipher (shift=13)
    res = try_caesar(s, shift=13)
    if res:
        results['Caesar Cipher (ROT13)'] = res

    # XOR with 0xFF
    res = try_xor(s, key=0xFF)
    if res:
        results['XOR with 0xFF'] = res

    # Binary Decode
    res = try_bin(s)
    if res:
        results['Binary Decode'] = res

    # Octal Decode
    res = try_oct(s)
    if res:
        results['Octal Decode'] = res

    # Punycode Decode
    res = try_punycode(s)
    if res:
        results['Punycode Decode'] = res

    # Brotli Decompress
    res = try_brotli(s)
    if res:
        results['Brotli Decompress'] = res

    # LZMA Decompress
    res = try_lzma(s)
    if res:
        results['LZMA Decompress'] = res

    return results

def main():
    if len(sys.argv) != 2:
        print("사용법: decoder {디코딩할 문자열}")
        sys.exit(1)

    input_str = sys.argv[1]
    print(f"원본 문자열:\n{input_str}\n")
    print("디코딩 시도 중...\n")

    decodings = try_all_decodings(input_str)

    if not decodings:
        print("알려진 인코딩 방식으로는 디코딩되지 않았습니다.")
    else:
        for method, result in decodings.items():
            try:
                decoded_str = result.decode('utf-8')
            except UnicodeDecodeError:
                decoded_str = result
            except AttributeError:
                decoded_str = result
            print(f"=== {method} 디코딩 결과 ===")
            print(decoded_str)
            print("------------------------------")

if __name__ == "__main__":
    main()

