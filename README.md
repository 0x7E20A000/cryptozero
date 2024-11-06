# Decoder

다양한 문자열 인코딩을 해제하고 분석하는 Python 스크립트

## 주요 기능
  - 다양한 디코딩: Base64, URL 인코딩, ROT13, Hex, Binary, JWT, Morse 등 지원
	- 문자열 분석: 엔트로피, 빈도 분석, 출력 가능 문자 비율 계산
  - CTF 패턴 탐지: flag{}, SQL 인젝션, XSS 등 흔한 패턴 탐지
	- 시각화: ASCII 그래프로 빈도, 엔트로피 시각화

# 설치 및 실행
`python3 decoder.py "디코딩할 문자열"`

# 사용 예시:"SGVsbG8gd29ybGQh"
`python3 decoder.py "SGVsbG8gd29ybGQh"`
```
Decoding Results Summary
===========================
Method      Entropy  Result
---------------------------
Original    3.92     SGVsbG8gd29ybGQh
Base64      4.80     Hello world!
```
