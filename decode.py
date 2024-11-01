#!/usr/bin/env python3
import sys
import base64
import zlib
import urllib.parse
import codecs
import json
import html
import math
import collections
from typing import Optional, Dict, Any, Union, Callable, Tuple
from dataclasses import dataclass
from functools import wraps
import logging

# 로깅 설정
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StringAnalysis:
    """문자열 분석 결과를 저장하는 데이터 클래스"""
    entropy: float
    character_frequencies: Dict[str, float]
    byte_frequencies: Dict[int, float]
    printable_ratio: float
    unique_chars: int
    length: int
    is_binary: bool

@dataclass
class DecodingResult:
    """디코딩 결과를 저장하는 데이터 클래스"""
    success: bool
    data: Optional[Union[str, bytes]] = None
    error: Optional[str] = None
    analysis: Optional[StringAnalysis] = None

class StringAnalyzer:
    """문자열 분석을 위한 클래스"""
    
    @staticmethod
    def calculate_entropy(data: Union[str, bytes]) -> float:
        """Shannon 엔트로피 계산"""
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        frequencies = collections.Counter(data)
        length = len(data)
        
        entropy = 0
        for count in frequencies.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
            
        return entropy

    @staticmethod
    def get_printable_ratio(data: Union[str, bytes]) -> float:
        """출력 가능한 문자의 비율 계산"""
        if isinstance(data, bytes):
            try:
                data = data.decode('utf-8')
            except UnicodeDecodeError:
                return 0.0
                
        printable = sum(1 for c in data if c.isprintable())
        return printable / len(data) if data else 0.0

    @staticmethod
    def get_character_frequencies(data: Union[str, bytes]) -> Dict[str, float]:
        """문자 빈도 분석"""
        if isinstance(data, bytes):
            try:
                data = data.decode('utf-8')
            except UnicodeDecodeError:
                return {}
                
        frequencies = collections.Counter(data)
        total = len(data)
        return {char: count/total for char, count in frequencies.most_common()}

    @staticmethod
    def get_byte_frequencies(data: Union[str, bytes]) -> Dict[int, float]:
        """바이트 빈도 분석"""
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        frequencies = collections.Counter(data)
        total = len(data)
        return {byte: count/total for byte, count in frequencies.most_common()}

    @staticmethod
    def analyze(data: Union[str, bytes]) -> StringAnalysis:
        """문자열 종합 분석"""
        is_binary = isinstance(data, bytes)
        if is_binary:
            try:
                str_data = data.decode('utf-8')
                is_binary = False
            except UnicodeDecodeError:
                str_data = None
        else:
            str_data = data

        return StringAnalysis(
            entropy=StringAnalyzer.calculate_entropy(data),
            character_frequencies=StringAnalyzer.get_character_frequencies(data),
            byte_frequencies=StringAnalyzer.get_byte_frequencies(data),
            printable_ratio=StringAnalyzer.get_printable_ratio(data),
            unique_chars=len(set(data if is_binary else str_data)),
            length=len(data),
            is_binary=is_binary
        )

def decode_handler(func: Callable) -> Callable:
    """디코딩 함수를 위한 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> DecodingResult:
        try:
            result = func(*args, **kwargs)
            if result:
                analysis = StringAnalyzer.analyze(result)
                return DecodingResult(success=True, data=result, analysis=analysis)
            return DecodingResult(success=False)
        except Exception as e:
            return DecodingResult(success=False, error=str(e))
    return wrapper

class VisualizationHelper:
    """분석 결과 시각화를 위한 도우미 클래스"""
    
    @staticmethod
    def create_histogram(frequencies: Dict[Union[str, int], float], max_width: int = 50) -> str:
        """빈도를 ASCII 히스토그램으로 변환"""
        if not frequencies:
            return "데이터 없음"
            
        max_freq = max(frequencies.values())
        result = []
        
        for char, freq in frequencies.items():
            if isinstance(char, int):
                char_repr = f"0x{char:02x}"
            else:
                char_repr = repr(char)[1:-1] if len(char) == 1 else char
                
            bar_length = int((freq / max_freq) * max_width)
            percentage = freq * 100
            bar = "█" * bar_length
            result.append(f"{char_repr:>8} [{bar:<{max_width}}] {percentage:>6.2f}%")
            
        return "\n".join(result)

    @staticmethod
    def create_entropy_meter(entropy: float, max_width: int = 50) -> str:
        """엔트로피를 시각적 미터로 표시"""
        max_entropy = 8.0  # 바이트당 최대 엔트로피
        meter_length = int((entropy / max_entropy) * max_width)
        meter = "█" * meter_length + "░" * (max_width - meter_length)
        return f"엔트로피: [{meter}] {entropy:.2f} bits/byte"

    @staticmethod
    def create_printable_meter(ratio: float, max_width: int = 50) -> str:
        """출력 가능한 문자 비율을 시각적 미터로 표시"""
        meter_length = int(ratio * max_width)
        meter = "█" * meter_length + "░" * (max_width - meter_length)
        return f"출력 가능 문자: [{meter}] {ratio*100:.1f}%"

class Decoder:
    """개선된 다중 포맷 디코더 클래스"""
    
    def __init__(self):
        self.MORSE_CODE = {
            'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
            'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
            'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
            'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
            'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
            'Z': '--..', '0': '-----', '1': '.----', '2': '..---', '3': '...--',
            '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..', 
            '9': '----.', ' ': '/', '.': '.-.-.-', ',': '--..--', '?': '..--..',
            '!': '-.-.--', '/': '-..-.', '(': '-.--.', ')': '-.--.-', '&': '.-...',
            ':': '---...', ';': '-.-.-.', '=': '-...-', '+': '.-.-.', '-': '-....-',
            '_': '..--.-', '"': '.-..-.', '\'': '...-..-', '@': '.--.-.'
        }
        self.REVERSE_MORSE = {v: k for k, v in self.MORSE_CODE.items()}
        self.viz = VisualizationHelper()

    @decode_handler
    def base64_decode(self, s: str) -> Optional[bytes]:
        """Base64 디코딩"""
        padding = '=' * (-len(s) % 4)
        return base64.b64decode(s + padding, validate=True)

    @decode_handler
    def base64url_decode(self, s: str) -> Optional[bytes]:
        """URL-safe Base64 디코딩"""
        padding = '=' * (-len(s) % 4)
        return base64.urlsafe_b64decode(s + padding)

    @decode_handler
    def zlib_decompress(self, data: bytes) -> Optional[bytes]:
        """ZLIB 압축 해제"""
        return zlib.decompress(data)

    @decode_handler
    def url_decode(self, s: str) -> str:
        """URL 디코딩"""
        decoded = urllib.parse.unquote_plus(s)
        return decoded if decoded != s else None

    @decode_handler
    def rot13_decode(self, s: str) -> str:
        """ROT13 디코딩"""
        decoded = codecs.decode(s, 'rot_13')
        return decoded if decoded != s else None

    @decode_handler
    def jwt_decode(self, s: str) -> str:
        """JWT 토큰 디코딩"""
        parts = s.split('.')
        if len(parts) != 3:
            return None
        
        header, payload, signature = parts
        decoded_header = self.base64url_decode(header).data
        decoded_payload = self.base64url_decode(payload).data
        
        if not decoded_header or not decoded_payload:
            return None
            
        try:
            header_json = json.loads(decoded_header)
            payload_json = json.loads(decoded_payload)
            
            return (f"Header:\n{json.dumps(header_json, indent=2)}\n\n"
                   f"Payload:\n{json.dumps(payload_json, indent=2)}\n\n"
                   f"Signature:\n{signature}")
        except json.JSONDecodeError:
            return None

    @decode_handler
    def morse_decode(self, s: str) -> str:
        """모스 부호 디코딩"""
        words = s.strip().split('/')
        decoded_words = []
        
        for word in words:
            letters = word.strip().split()
            decoded_letters = [self.REVERSE_MORSE.get(letter, '') for letter in letters]
            if not all(decoded_letters):  # 유효하지 않은 모스 부호가 있으면 None 반환
                return None
            decoded_words.append(''.join(decoded_letters))
            
        decoded = ' '.join(decoded_words)
        return decoded if decoded != s else None

    @decode_handler
    def caesar_decode(self, s: str, shift: int = 13) -> str:
        """시저 암호 디코딩"""
        def shift_char(c: str, shift: int) -> str:
            if c.isalpha():
                base = ord('A') if c.isupper() else ord('a')
                return chr((ord(c) - base + shift) % 26 + base)
            return c
            
        decoded = ''.join(shift_char(c, shift) for c in s)
        return decoded if decoded != s else None

    @decode_handler
    def json_decode(self, s: str) -> str:
        """JSON 디코딩"""
        obj = json.loads(s)
        return json.dumps(obj, indent=2, ensure_ascii=False)

    def decode_all(self, input_str: str) -> Dict[str, DecodingResult]:
        """모든 가능한 디코딩 시도"""
        results = {}
        
        # 입력 문자열 분석
        input_analysis = StringAnalyzer.analyze(input_str)
        results['Original'] = DecodingResult(success=True, data=input_str, analysis=input_analysis)
        
        # 기본 디코딩
        decoders = {
            'Base64': self.base64_decode,
            'Base64 URL-safe': self.base64url_decode,
            'URL': self.url_decode,
            'ROT13': self.rot13_decode,
            'JWT': self.jwt_decode,
            'Morse': self.morse_decode,
            'Caesar (ROT13)': lambda x: self.caesar_decode(x, 13),
            'JSON': self.json_decode
        }

        for name, decoder in decoders.items():
            result = decoder(input_str)
            if result.success:
                results[name] = result
                
                # zlib 압축 해제 시도 (바이너리 결과에 대해)
                if isinstance(result.data, bytes):
                    zlib_result = self.zlib_decompress(result.data)
                    if zlib_result.success:
                        results[f'{name} + ZLIB'] = zlib_result

        return results

    def format_output(self, method: str, result: DecodingResult) -> str:
        """디코딩 결과 포맷팅과 시각화"""
        output = []
        
        # 성공한 디코딩 결과만 표시
        if not result.success:
            return ""
            
        # 간단한 구분선으로 변경
        output.append(f"\n[{method}]")
        
        # 데이터 출력을 더 실용적으로
        if isinstance(result.data, bytes):
            try:
                decoded_str = result.data.decode('utf-8')
                output.append(f"Result: {decoded_str}")
                output.append(f"Hex: {result.data.hex()[:50]}..." if len(result.data) > 25 else result.data.hex())
            except UnicodeDecodeError:
                output.append(f"Binary ({len(result.data)} bytes)")
                output.append(f"Hex: {result.data.hex()[:50]}...")
        else:
            output.append(f"Result: {result.data}")

        # 중요 메트릭만 간단히 표시
        if result.analysis:
            output.append(f"Length: {result.analysis.length} | Entropy: {result.analysis.entropy:.2f} | Unique chars: {result.analysis.unique_chars}")
            
            # 특이사항이 있는 경우만 빈도 분석 표시
            if result.analysis.entropy > 4.0 or result.analysis.unique_chars < 5:
                output.append("\nFrequency Analysis (Top 5):")
                freq_data = dict(list(
                    (result.analysis.byte_frequencies if result.analysis.is_binary 
                    else result.analysis.character_frequencies).items())[:5]
                )
                output.append(self.viz.create_histogram(freq_data, max_width=30))

        output.append("-" * 50)
        return "\n".join(output)

def main():
    if len(sys.argv) != 2:
        print("사용법: decoder.py <디코딩할 문자열>")
        sys.exit(1)

    input_str = sys.argv[1]
    decoder = Decoder()
    
    print(f"입력 문자열 분석 시작...")
    results = decoder.decode_all(input_str)
    
    if len(results) == 1:  # 원본만 있는 경우
        print("\n알려진 인코딩 방식으로는 디코딩되지 않았습니다.")
    
    for method, result in results.items():
        print(decoder.format_output(method, result))

if __name__ == "__main__":
    main()
