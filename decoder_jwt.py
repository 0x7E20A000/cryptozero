#!/usr/bin/env python3
import sys
import base64
import zlib
import codecs
import urllib.parse
import json

def base64_url_decode(s):
    try:
        padding = '=' * (-len(s) % 4)
        return base64.urlsafe_b64decode(s + padding)
    except Exception as e:
        print(f"Base64 URL-safe 디코딩 실패: {e}")
        return None

def zlib_decompress(data):
    try:
        return zlib.decompress(data).decode('utf-8')
    except Exception as e:
        print(f"zlib 압축 해제 실패: {e}")
        return None

def decode_jwt(jwt_token):
    parts = jwt_token.split('.')
    if len(parts) != 3:
        print("유효한 JWT 토큰이 아닙니다.")
        return

    header_b64, payload_b64, signature_b64 = parts

    print("=== Header 디코딩 ===")
    header_decoded = base64_url_decode(header_b64)
    if header_decoded:
        try:
            header_json = json.loads(header_decoded)
            print(json.dumps(header_json, indent=4, ensure_ascii=False))
        except json.JSONDecodeError:
            print(header_decoded.decode('utf-8', errors='replace'))
    print("------------------------------")

    print("=== Payload 디코딩 ===")
    payload_decoded = base64_url_decode(payload_b64)
    if payload_decoded:
        # 시도: zlib 압축 해제
        if payload_decoded.startswith(b'x\x9c'):
            print("Payload가 zlib로 압축되어 있습니다. 압축 해제 시도:")
            decompressed = zlib_decompress(payload_decoded)
            if decompressed:
                print(decompressed)
        else:
            try:
                payload_json = json.loads(payload_decoded)
                print(json.dumps(payload_json, indent=4, ensure_ascii=False))
            except json.JSONDecodeError:
                print(payload_decoded.decode('utf-8', errors='replace'))
    print("------------------------------")

    print("=== Signature ===")
    print(signature_b64)
    print("------------------------------")

def main():
    if len(sys.argv) != 2:
        print("사용법: decoder {디코딩할 JWT 토큰}")
        sys.exit(1)

    jwt_token = sys.argv[1]
    print(f"원본 JWT 토큰:\n{jwt_token}\n")
    print("디코딩 시도 중...\n")

    decode_jwt(jwt_token)

if __name__ == "__main__":
    main()
