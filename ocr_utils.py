import re

class OCRUtils:
    
    MISREAD_MAPPING = {
        "O": "0",
        "o": "0",
        "A": "4",
    }
    
    @staticmethod
    def concat_strings(str1, str2):
        return str(str1) + str(str2)
    
    @staticmethod
    def apply_misread_mapping(text):
        return "".join(OCRUtils.MISREAD_MAPPING.get(ch, ch) for ch in text).strip()
    
    @staticmethod
    def extract_numeric_part(token):
        token = token.strip()
        combined_match = re.search(r'(\d+)[,.](\d+)', token)
        if combined_match:
            int_part, frac_part = combined_match.groups()
            return ('combined', int_part, frac_part)
        
        integer_or_trailing_match = re.fullmatch(r'\d+[.,]?', token)
        if integer_or_trailing_match:
            cleaned_int = token.rstrip('.,')
            if cleaned_int.isdigit():
                if len(cleaned_int) == 2:
                    return ('fraction_as_int', None, cleaned_int)
                else:
                    return ('integer', cleaned_int, None)

        fraction_match = re.fullmatch(r'[.,]\d+', token)
        if fraction_match:
            return ('fraction', None, token)
        
        return None
    
    @staticmethod
    def is_valid_format(final_string):
        digits_only = final_string.replace('.', '').replace(',', '')
        return len(digits_only) <= 3
    
    @staticmethod
    def get_max_height_sign(raw_result):
        MIN_CONFIDENCE = 0.20
        SINGLE_COMBINED_THRESHOLD = 0.80
        INT_FRAC_CONF_THRESHOLD = 0.40
        
        combined_candidates = []
        integer_candidates = []
        fraction_candidates = []
        tokens_used = []
        
        for line in raw_result:
            for word in line:
                raw_text, conf = word[1]
                x_coord = word[0][0][0]
                box = word[0]
                if conf < MIN_CONFIDENCE:
                    continue
                cleaned = OCRUtils.apply_misread_mapping(raw_text)
                result = OCRUtils.extract_numeric_part(cleaned)
                if result is None:
                    continue
                ttype, int_part, frac_part = result
                tokens_used.append((cleaned, conf, x_coord, box))
                if ttype == 'combined':
                    normalized = f"{int_part}.{frac_part}"
                    combined_candidates.append((normalized, conf, x_coord, box))
                elif ttype == 'integer':
                    integer_candidates.append((int_part, conf, x_coord, box))
                elif ttype == 'fraction':
                    norm_frac = frac_part if frac_part else cleaned.replace(',', '.')
                    if not norm_frac.startswith('.'):
                        norm_frac = '.' + norm_frac.lstrip(',.')
                    fraction_candidates.append((norm_frac, conf, x_coord, box))
                elif ttype == 'fraction_as_int':
                    norm_frac = '.' + frac_part.lstrip(',.')
                    fraction_candidates.append((norm_frac, conf, x_coord, box))
        
        if not tokens_used:
            return "UNDETECTED", []
        
        valid_combined = [c for c in combined_candidates if c[1] >= SINGLE_COMBINED_THRESHOLD]
        if valid_combined:
            best_combined = max(valid_combined, key=lambda x: x[1])
            if OCRUtils.is_valid_format(best_combined[0]):
                return best_combined[0], [best_combined]
            else:
                all_candidates = combined_candidates + integer_candidates + fraction_candidates
                return "UNCLASSIFIED", all_candidates
        else:
            if len(combined_candidates) == 1:
                all_candidates = combined_candidates + integer_candidates + fraction_candidates
                return "UNCLASSIFIED", all_candidates
        
        integer_candidates.sort(key=lambda x: x[1], reverse=True)
        fraction_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if integer_candidates and fraction_candidates:
            best_int = integer_candidates[0]
            best_frac = fraction_candidates[0]
            if best_int[1] < INT_FRAC_CONF_THRESHOLD or best_frac[1] < INT_FRAC_CONF_THRESHOLD:
                all_candidates = combined_candidates + integer_candidates + fraction_candidates
                return "UNCLASSIFIED", all_candidates
            
            combined_value = f"{best_int[0]}.{best_frac[0].lstrip('.')}"
            if OCRUtils.is_valid_format(combined_value):
                return combined_value, [best_int, best_frac]
            else:
                all_candidates = combined_candidates + integer_candidates + fraction_candidates
                return "UNCLASSIFIED", all_candidates
        
        all_candidates = combined_candidates + integer_candidates + fraction_candidates
        return "UNCLASSIFIED", all_candidates
