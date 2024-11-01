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
from typing import Optional, Dict, Any, Union, Callable, Tuple, List
from dataclasses import dataclass
from functools import wraps
import logging
import re

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

# Colors 클래스는 파일 상단에 추가
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'    # 흐리게
    RESET = '\033[0m'
    
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

        # 기존 코드 유지
        self.common_patterns = {
            'php_magic': r'<\?php|eval|system|exec|passthru|shell_exec',
            'sql_injection': r'UNION|SELECT|FROM|WHERE|CONCAT|GROUP_BY',
            'xss': r'<script|javascript:|onerror=|onload=',
            'path_traversal': r'\.\.\/|\.\.\\',
            'command_injection': r';ls|;cat|;pwd|;id|;whoami',
            'hex_encoded': r'\\x[0-9a-fA-F]{2}',
            'base64_pattern': r'^[A-Za-z0-9+/]*={0,2}$',
            'jwt_pattern': r'^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]*$'
        }
        
        # 일반적인 CTF 플래그 포맷 패턴
        self.flag_patterns = [
            r'flag{.*}',
            r'CTF{.*}',
            r'KEY{.*}',
            r'picoCTF{.*}',
            r'FLAG_[a-zA-Z0-9]{16,}',
            r'[a-fA-F0-9]{32}'  # MD5 해시 패턴
        ]

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

    def detect_patterns(self, data: Union[str, bytes]) -> List[str]:
        """알려진 패턴 탐지"""
        if isinstance(data, bytes):
            try:
                data = data.decode('utf-8', errors='ignore')
            except:
                return []

        findings = []
        # 패턴 검사
        for pattern_name, pattern in self.common_patterns.items():
            if re.search(pattern, data, re.IGNORECASE):
                findings.append(f"Found {pattern_name} pattern")
        
        # 플래그 포맷 검사
        for pattern in self.flag_patterns:
            match = re.search(pattern, data)
            if match:
                findings.append(f"Possible flag found: {match.group(0)}")
        
        return findings

    @decode_handler
    def hex_decode(self, s: str) -> Optional[bytes]:
        """16진수 디코딩"""
        try:
            # 0x 또는 \x 형식 모두 처리
            s = s.replace('0x', '').replace('\\x', '').replace(' ', '')
            return bytes.fromhex(s)
        except:
            return None

    @decode_handler
    def binary_decode(self, s: str) -> Optional[str]:
        """2진수 문자열 디코딩"""
        try:
            # 공백과 0b 제거
            s = s.replace(' ', '').replace('0b', '')
            # 8비트 단위로 분할
            chunks = [s[i:i+8] for i in range(0, len(s), 8)]
            return ''.join(chr(int(chunk, 2)) for chunk in chunks)
        except:
            return None

    @decode_handler
    def reverse_string(self, s: str) -> str:
        """문자열 뒤집기"""
        return s[::-1]

    def decode_all(self, input_str: str) -> Dict[str, DecodingResult]:
        """모든 가능한 디코딩 시도"""
        results = {}
        
        # 입력 문자열 분석
        input_analysis = StringAnalyzer.analyze(input_str)
        results['Original'] = DecodingResult(success=True, data=input_str, analysis=input_analysis)
        
        # 패턴 탐지 결과 추가
        patterns = self.detect_patterns(input_str)
        if patterns:
            results['Pattern Detection'] = DecodingResult(
                success=True, 
                data='\n'.join(patterns)
            )
        
        # CTF용 디코더 추가
        decoders = {
            'Base64': self.base64_decode,
            'Base64 URL-safe': self.base64url_decode,
            'URL': self.url_decode,
            'ROT13': self.rot13_decode,
            'Hex': self.hex_decode,
            'Binary': self.binary_decode,
            'Reverse': self.reverse_string,
            'JWT': self.jwt_decode,
            'Morse': self.morse_decode,
            'Caesar (ROT13)': lambda x: self.caesar_decode(x, 13),
            'JSON': self.json_decode
        }

        for name, decoder in decoders.items():
            result = decoder(input_str)
            if result.success:
                results[name] = result

        # 결과 테이블 출력
        self._print_summary_table(results)
        
        return results

    def _print_summary_table(self, results: Dict[str, DecodingResult]) -> None:
        """결과 요약 테이블 출력"""
        print(f"\n{Colors.BOLD}Decoding Results Summary{Colors.RESET}")
        print("=" * 70)
        print(f"{Colors.BOLD}{'Method':<15} {'Length':>8} {'Entropy':>8} {'Result Preview':<30}{Colors.RESET}")
        print("-" * 70)
        
        for method, result in results.items():
            if not result.success or method == 'Pattern Detection':
                continue
                
            if result.analysis:
                if result.analysis.entropy > 6.0:
                    entropy_color = Colors.RED
                elif result.analysis.entropy > 4.0:
                    entropy_color = Colors.YELLOW
                else:
                    entropy_color = Colors.GREEN
                    
                preview = str(result.data)[:30]
                if len(str(result.data)) > 30:
                    preview += "..."
                    
                print(
                    f"{Colors.CYAN}{method:<15}{Colors.RESET} "
                    f"{result.analysis.length:>8} "
                    f"{entropy_color}{result.analysis.entropy:>8.2f}{Colors.RESET} "
                    f"{Colors.DIM}{preview:<30}{Colors.RESET}"
                )
        
        print("-" * 70)

    def format_output(self, method: str, result: DecodingResult) -> str:
        """디코딩 결과 포맷팅"""
        if not result.success or method == 'Pattern Detection':
            return ""
        
        output = []
        
        # 1. 분석 보고서 (Original만)
        if method == 'Original':
            output.append("\nAnalysis Report")
            output.append("=" * 50)
            
        output.append(f"\n[{method}]")
        
        # 2. 핵심 데이터
        if isinstance(result.data, bytes):
            try:
                decoded_str = result.data.decode('utf-8')
                output.append(f"▶ {decoded_str}")
            except UnicodeDecodeError:
                output.append(f"▶ Binary data ({len(result.data)} bytes)")
        else:
            output.append(f"▶ {result.data}")

        # 3. Original일 때만 상세 분석
        if method == 'Original' and result.analysis:
            # 기본 메트릭
            output.append(f"\nLength: {result.analysis.length} | Unique chars: {result.analysis.unique_chars}")
            
            # 엔트로피 시각화
            entropy_bar = "▁▂▃▄▅▆▇█"[int((result.analysis.entropy / 8.0) * 7)]
            output.append(f"Entropy: {entropy_bar} {result.analysis.entropy:.2f}/8.0")
            
            # 알파벳 분포 시각화 (A-Z 또는 a-z가 있는 경우만)
            char_freq = result.analysis.character_frequencies
            alpha_dist = {c: char_freq.get(c, 0) for c in 'abcdefghijklmnopqrstuvwxyz'}
            if any(alpha_dist.values()):
                output.append("\nCharacter distribution (a-z):")
                max_freq = max(alpha_dist.values()) if alpha_dist.values() else 1
                dist_line = ""
                for c, freq in alpha_dist.items():
                    if freq > 0:
                        height = int((freq / max_freq) * 7)
                        dist_line += "▁▂▃▄▅▆▇█"[height]
                    else:
                        dist_line += "."
                output.append(dist_line)
                output.append("abcdefghijklmnopqrstuvwxyz")

        # 4. 패턴 매칭 결과
        patterns = self.detect_patterns(result.data)
        if patterns:
            output.append("\nDetected:")
            output.append("  " + "\n  ".join(patterns))

        # 5. 특이사항
        if result.analysis:
            notes = []
            if result.analysis.entropy > 4.0:
                notes.append("High entropy detected")
            if result.analysis.unique_chars < 5:
                notes.append("Low character variety")
            if notes:
                output.append("\nNotes: " + " | ".join(notes))

        output.append("-" * 50)
        return "\n".join(output)

def format_summary_table(self, results: Dict[str, DecodingResult]) -> str:
    """모든 디코딩 결과를 테이블로 표시"""
    output = []
    output.append(f"\n{Colors.BOLD}Decoding Results Summary{Colors.RESET}")
    output.append("=" * 70)
    output.append(f"{Colors.BOLD}{'Method':<15} {'Length':>8} {'Entropy':>8} {'Result Preview':<30}{Colors.RESET}")
    output.append("-" * 70)
    
    for method, result in results.items():
        if not result.success or method == 'Pattern Detection':
            continue
            
        # 엔트로피에 따른 색상 선택
        if result.analysis:
            if result.analysis.entropy > 6.0:
                entropy_color = Colors.RED
            elif result.analysis.entropy > 4.0:
                entropy_color = Colors.YELLOW
            else:
                entropy_color = Colors.GREEN
                
            preview = str(result.data)[:30]
            if len(str(result.data)) > 30:
                preview += "..."
                
            output.append(
                f"{Colors.CYAN}{method:<15}{Colors.RESET} "
                f"{result.analysis.length:>8} "
                f"{entropy_color}{result.analysis.entropy:>8.2f}{Colors.RESET} "
                f"{Colors.DIM}{preview:<30}{Colors.RESET}"
            )
    
    output.append("-" * 70)
    return "\n".join(output)

def decode_all(self, input_str: str) -> Dict[str, DecodingResult]:
    results = super().decode_all(input_str)
    print(self.format_summary_table(results))  # 테이블 출력 추가
    return results

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
