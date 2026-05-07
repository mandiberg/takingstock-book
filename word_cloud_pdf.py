"""
make_wordcloud_pdf.py
Tench Cholnoky · macOS Sequoia 15.5 · Python 3.10

requires the gensim conda env - gensim311 for MM
"""

import os, tempfile, math, datetime
import pandas as pd
import random
import numpy as np
import csv
from wordcloud import WordCloud                          
from PIL import ImageCms
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import cm
from pick import pick

try:
    import pikepdf
except ImportError:
    pikepdf = None


import gensim
from gensim import corpora
from nltk.stem import WordNetLemmatizer, SnowballStemmer

# ---------- CONFIG -----------------------------------------------------------

GLOBAL_PATH = os.path.dirname(os.path.abspath(__file__))
TAKINGSTOCK_PATH = os.path.join(os.path.dirname(GLOBAL_PATH), 'facemap/')

INPUT_PATH  = os.path.join(GLOBAL_PATH, "input_csvs/word_cloud/")
MODEL_PATH =  os.path.join(GLOBAL_PATH, "model/")
STOPWORD_PATH = os.path.join(TAKINGSTOCK_PATH, "model_files/")
OUTPUT_PATH = os.path.join(GLOBAL_PATH, 'outputs/word_cloud/')
PASSED_WORDS_POS_FILE = os.path.join(GLOBAL_PATH, "passed_words_pos.csv")
STOPWORD_DF_PATH = os.path.join(GLOBAL_PATH, "topic_word_stopword.csv")
FOOTER_FILE = os.path.join(GLOBAL_PATH, "footers.csv")
ICC_PROFILE_PATH = os.path.join(GLOBAL_PATH, "printing/PSOcoated_v3.icc")

# print(f"Paths: Input: {INPUT_PATH}, Model: {MODEL_PATH}, Stopwords: {STOPWORD_PATH}, Output: {OUTPUT_PATH}")

PDF_DATA = {}

FONT_NAME   = "Cardo"
FONT_FILE   = os.path.join(GLOBAL_PATH, "fonts/Cardo.ttf") 
FOOTER_FONT_FILE = os.path.join(GLOBAL_PATH, "fonts/IBMPlexMono-SemiBold.ttf")
FOOTER_FONT_NAME = "IBMPlexMono-SemiBold"
PAGE_SIZE   = [20*cm, 30*cm]  # 20cm × 30cm (ReportLab uses points; cm converts to pt)

QUALITY = 95  # JPEG quality for CMYK images (0-100)

# Toggle prepress marks/bleed canvas on or off.
ENABLE_BLEED_AND_CROPS = True

# Prepress: bleed, crop marks, slug (all in points; 1 mm = 72/25.4 pt)
_MM          = 72 / 25.4
BLEED        = 3  * _MM   # 3 mm bleed all sides
MARK_OFFSET  = 5  * _MM   # 5 mm from bleed edge to near end of mark
MARK_LENGTH  = 3  * _MM   # 3 mm long crop marks
MARK_WEIGHT  = 0.125       # pt stroke weight
SLUG         = (BLEED + MARK_OFFSET + MARK_LENGTH) if ENABLE_BLEED_AND_CROPS else 0
CANVAS_SIZE  = [PAGE_SIZE[0] + 2 * SLUG, PAGE_SIZE[1] + 2 * SLUG]

# Scale factor: original page was 432×648 pt (6×9"); new is 20×30cm. Scale ≈ 1.31.
_ORIG_PAGE_WIDTH = 432
PAGE_SCALE = (20*cm) / _ORIG_PAGE_WIDTH
print(f"PAGE_SCALE is {PAGE_SCALE}")

# Margin and gutter settings (in points: 1" = 72 pt)
OUTER_MARGIN = int(0.625 * 72)   # 5/8" outside
INNER_MARGIN = int(0.75 * 72)  # 0.75" inside (binding)
TOP_MARGIN = int(0.5 * 72)     # 0.5" top
BOTTOM_MARGIN = int(1.0 * 72)  # 1" bottom (footer centered in this band)

# Footer settings
FOOTER_TEXT = "Topic "  # Base text, topic number will be added dynamically
FOOTER_FONT_SIZE = int(9 * PAGE_SCALE)
FOOTER_NUDGE = 20  # Points to nudge footer up (positive) or down (negative)

# Cache for word colors
_word_color_cache = {}

# List to track words that pass/clear
passed_words_list = []

# List to track words without POS tags (trigger POS_COLOR_OTHER)
words_without_pos_list = []

SIDE = "left"

#batch Processing
BATCH_PROCESS = True
PROCESS_SELECT = [18,19]
CSV_LIST = {}

#cutoff for how many rows of the CSV to add to the textcloud
CUTOFF = True
NUM_ROWS = 100
print(f"CUTOFF is {CUTOFF}")
print("NUM_ROWS is", NUM_ROWS)
MANUAL_PICK = False
print(f"MANUAL PICK IS {MANUAL_PICK}")


PURPLE_COLOR = "rgb(139, 131, 187)"
# GREEN_COLOR = "rgb(192, 241, 194)"
# BLUE_COLOR = "rgb(192, 194, 241)"
PINK_COLOR = "rgb(225, 181, 190)"
CYAN_COLOR = "rgb(109, 177, 212)"
NEON_CYAN_COLOR = "rgb(79, 199, 208)"
LIGHT_CYAN_COLOR =  "rgb(148, 194, 219)"
RED_COLOR = "rgb(176, 58, 54)"
MAGENTA_COLOR = "rgb(200, 95, 125)"
YELLOW_COLOR = "rgb(240, 207, 99)"
BLACK_COLOR = "rgb(0,0,0)"
DARK_GRAY_COLOR = "rgb(70,70,70)"
GRAY_COLOR = "rgb(215,215,215)"
LIGHT_GRAY_COLOR = "rgb(235,235,235)"
LIGHT_GRAY_COLOR_2 = "rgb(230,230,230)"
LIGHT_GRAY_COLOR_3 = "rgb(225,225,225)"
LIGHT_GRAY_COLOR_4 = "rgb(222,222,222)"
WHITE_COLOR = "rgb(255,255,255)"

YELLOW_COLOR = "rgb(240, 207, 99)"
# YELLOW_COLOR_1 = "rgb(221,191,90)"
# YELLOW_COLOR_2 = "rgb(224,190,78)"
# YELLOW_COLOR_3 = "rgb(237,202,90)"



# Word-cloud cosmetics (scaled for 20×30cm page)
FONT_MIN = int(23)         # adjust to taste
FONT_MAX = int(1100)       # Maximum font size - will be scaled per topic based on global proportions
WC_WIDTH, WC_HEIGHT = int(3200 * PAGE_SCALE), int(4800 * PAGE_SCALE)  # px; higher = sharper
STOPWORD_COLOR = LIGHT_GRAY_COLOR
WORD_COLOR = BLACK_COLOR
BACKGROUND_COLOR = WHITE_COLOR

POS_COLOR_NOUN = WORD_COLOR
POS_COLOR_ADJECTIVE = NEON_CYAN_COLOR
POS_COLOR_VERB = YELLOW_COLOR
POS_COLOR_OTHER = WORD_COLOR

print(f"WORD_COLOR is {WORD_COLOR}")
print(f"STOPWORD_COLOR is {STOPWORD_COLOR}")
print(f"BACKGROUND_COLOR is {BACKGROUND_COLOR}")
print(f"POS_COLOR_NOUN is {POS_COLOR_NOUN}")
print(f"POS_COLOR_ADJECTIVE is {POS_COLOR_ADJECTIVE}")
print(f"POS_COLOR_VERB is {POS_COLOR_VERB}")
print(f"POS_COLOR_OTHER is {POS_COLOR_OTHER}")

OUT_PDF     = os.path.join(OUTPUT_PATH, f"wordcloud_FULL_BATCH_{BATCH_PROCESS}_GRAY_{STOPWORD_COLOR}")  # final file
CMYK_ASSET_PATH = os.path.join(OUTPUT_PATH, "cmyk")
COLOR_MANAGED_CMYK = True
OUTPUT_INTENT_IDENTIFIER = "PSO Coated v3"
OUTPUT_INTENT_INFO = "PSO Coated v3 (FOGRA51)"

# Scaling configuration
SCALE_BUCKETS = [

    # {"name": "bucket1", "topic_count": 1,  "output_min": 0.95, "output_max": 1.00},
    # {"name": "bucket2", "topic_count": 1, "output_min": 0.80, "output_max": 0.95},
    # {"name": "bucket3", "topic_count": 1, "output_min": 0.60, "output_max": 0.80},
    # {"name": "bucket4", "topic_count": 1, "output_min": 0.40, "output_max": 0.60},
    # {"name": "bucket5", "topic_count": 1, "output_min": 0.20, "output_max": 0.40},
    # {"name": "bucket6", "topic_count": 9999, "output_min": 0.10, "output_max": 0.20},  # overflow bucket (currently has 5 topics)
    {"name": "bucket1", "topic_count": 10,  "output_min": 0.95, "output_max": 1.00},
    {"name": "bucket2", "topic_count": 10, "output_min": 0.80, "output_max": 0.95},
    {"name": "bucket3", "topic_count": 13, "output_min": 0.60, "output_max": 0.80},
    {"name": "bucket4", "topic_count": 13, "output_min": 0.40, "output_max": 0.60},
    {"name": "bucket5", "topic_count": 13, "output_min": 0.20, "output_max": 0.40},
    {"name": "bucket6", "topic_count": 9999, "output_min": 0.10, "output_max": 0.20},  # overflow bucket (currently has 5 topics)
]
# -----------------------------------------------------------------------------
def analyze_csv(input_csv, input_path, num_rows):
    df = pd.read_csv(input_path+input_csv)
    df = df.dropna()
    if CUTOFF:
        df = df.head(num_rows)
    return input_csv, df,


def load_pos_lookup(csv_path):
    """
    Load a word -> POS mapping from the passed_words_pos.csv file.
    """
    if not os.path.exists(csv_path):
        print(f"WARNING: POS tag file not found at {csv_path}. Default colors will be used.")
        return {}
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"WARNING: Unable to read POS tag file '{csv_path}': {exc}")
        return {}
    
    if 'word' not in df.columns or 'POS' not in df.columns:
        print(f"WARNING: POS tag file '{csv_path}' is missing required columns 'word' and/or 'POS'.")
        return {}
    
    df = df.dropna(subset=['word', 'POS'])
    df['word'] = df['word'].astype(str).str.strip()
    df['POS'] = df['POS'].astype(str).str.strip()
    
    lookup = {}
    for _, row in df.iterrows():
        key = row['word'].lower()
        if key:
            lookup[key] = row['POS']
    print(f"Loaded {len(lookup)} POS entries from {csv_path}")
    return lookup


PASSED_WORDS_POS_LOOKUP = load_pos_lookup(PASSED_WORDS_POS_FILE)


def build_freqs_with_replacements(df, topic_word_stopword_df):
    """
    Build frequency dictionary from dataframe and apply word replacements.
    
    Args:
        df: DataFrame with 'description' and 'count' columns
        topic_word_stopword_df: DataFrame with word replacement rules
    
    Returns:
        Dictionary mapping words to their counts (after replacements)
    """
    freqs = dict(zip(df["description"], df["count"]))
    
    # Preprocess: Replace words based on "Replace" column in topic_word_stopword_df
    # If "stopped" is empty and "Replace" has a value, replace the word
    if 'Replace' in topic_word_stopword_df.columns:
        words_to_replace = {}
        for idx, row in topic_word_stopword_df.iterrows():
            word = row['word']
            stopped_value = row.get('stopped', None)
            replace_value = row.get('Replace', '')
            
            # Check if stopped is empty (None, NaN, empty string) and Replace has a value
            if (pd.isna(stopped_value) or stopped_value == '' or stopped_value is None) and \
               pd.notna(replace_value) and replace_value != '':
                # If word exists in freqs, mark it for replacement
                if word in freqs:
                    words_to_replace[word] = replace_value
        
        # Perform replacements and merge counts if replace value already exists
        for old_word, new_word in words_to_replace.items():
            if old_word in freqs:
                count = freqs.pop(old_word)
                # If new_word already exists, add the counts together
                if new_word in freqs:
                    freqs[new_word] += count
                else:
                    freqs[new_word] = count
                print(f"Replaced '{old_word}' with '{new_word}' (count: {count})")
    
    return freqs


def compute_global_scale(all_freqs_dicts, use_log_scale=False):
    """
    Compute global min/max scaling across all frequency dictionaries.
    
    Args:
        all_freqs_dicts: List of frequency dictionaries from all CSVs
        use_log_scale: If True, uses log1p transform to reduce skew. If False, uses linear scaling.
    
    Returns:
        Dictionary with 'min' and 'max' values for scaling, and 'use_log' flag
    """
    all_values = []
    for freqs_dict in all_freqs_dicts:
        for count in freqs_dict.values():
            if count > 0:  # Only include positive values
                all_values.append(count)
    
    if not all_values:
        print("  WARNING: No frequency values found! Using fallback scale.")
        return {'min': 0, 'max': 1, 'use_log': False}
    
    print(f"  Collected {len(all_values)} frequency values from {len(all_freqs_dicts)} CSVs")
    print(f"  Raw values - min={min(all_values)}, max={max(all_values)}, mean={np.mean(all_values):.2f}, median={np.median(all_values):.2f}")
    
    if use_log_scale:
        # Apply log1p transform to reduce skew
        log_values = np.log1p(all_values)
        min_val = np.min(log_values)
        max_val = np.max(log_values)
        
        print(f"  After log1p - min={min_val:.4f}, max={max_val:.4f}, mean={np.mean(log_values):.4f}, median={np.median(log_values):.4f}")
        
        # Handle edge case where all values are the same
        if min_val == max_val:
            print(f"  WARNING: All log values are the same! Adding buffer.")
            return {'min': min_val, 'max': max_val + 0.1, 'use_log': True}
        
        return {'min': min_val, 'max': max_val, 'use_log': True}
    else:
        # Use linear scaling
        min_val = min(all_values)
        max_val = max(all_values)
        
        print(f"  Linear scaling - min={min_val:.2f}, max={max_val:.2f}, mean={np.mean(all_values):.2f}")
        
        # Handle edge case where all values are the same
        if min_val == max_val:
            print(f"  WARNING: All values are the same! Adding buffer.")
            return {'min': min_val, 'max': max_val + 1.0, 'use_log': False}
        
        return {'min': min_val, 'max': max_val, 'use_log': False}


def _clamp(value, min_value=0.0, max_value=1.0):
    """Clamp a numeric value into [min_value, max_value]."""
    return max(min_value, min(max_value, value))


def ensure_output_paths():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    if COLOR_MANAGED_CMYK:
        os.makedirs(CMYK_ASSET_PATH, exist_ok=True)


def get_cmyk_transform(icc_profile_path):
    if not os.path.exists(icc_profile_path):
        raise FileNotFoundError(f"ICC profile not found: {icc_profile_path}")

    srgb_profile = ImageCms.createProfile("sRGB")
    cmyk_profile = ImageCms.getOpenProfile(icc_profile_path)
    transform = ImageCms.buildTransformFromOpenProfiles(
        srgb_profile,
        cmyk_profile,
        "RGB",
        "CMYK",
        renderingIntent=0,
    )
    return transform


def save_cmyk_jpeg_from_wordcloud(wc, csv_number, cmyk_transform, icc_profile_path):
    rgb_image = wc.to_image().convert("RGB")
    cmyk_image = ImageCms.applyTransform(rgb_image, cmyk_transform)
    cmyk_path = os.path.join(CMYK_ASSET_PATH, f"topic_{csv_number}_cmyk.jpg")

    with open(icc_profile_path, "rb") as f:
        icc_bytes = f.read()

    cmyk_image.save(
        cmyk_path,
        format="JPEG",
        quality=QUALITY,
        subsampling=0,
        icc_profile=icc_bytes,
    )
    return cmyk_path, cmyk_image.size


def draw_print_marks(c, page_info_label, page_num):
    """Draw crop marks and slug info text outside the trim area per printer spec."""
    tw = PAGE_SIZE[0]
    th = PAGE_SIZE[1]
    s  = SLUG
    b  = BLEED
    mo = MARK_OFFSET
    ml = MARK_LENGTH

    c.saveState()
    c.setLineWidth(MARK_WEIGHT)
    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)

    # Horizontal marks (indicate left/right trim edges at each corner)
    lx1 = s - b - mo - ml
    lx2 = s - b - mo
    rx1 = s + tw + b + mo
    rx2 = s + tw + b + mo + ml

    # Vertical marks (indicate top/bottom trim edges at each corner)
    by1 = s - b - mo - ml
    by2 = s - b - mo
    ty1 = s + th + b + mo
    ty2 = s + th + b + mo + ml

    # Bottom-left corner
    c.line(lx1, s, lx2, s)
    c.line(s, by1, s, by2)
    # Bottom-right corner
    c.line(rx1, s, rx2, s)
    c.line(s + tw, by1, s + tw, by2)
    # Top-left corner
    c.line(lx1, s + th, lx2, s + th)
    c.line(s, ty1, s, ty2)
    # Top-right corner
    c.line(rx1, s + th, rx2, s + th)
    c.line(s + tw, ty1, s + tw, ty2)

    # Slug info line: filename | page | topic | date
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    slug_text = f"{os.path.basename(OUT_PDF)}  |  p.{page_num}  |  {page_info_label}  |  {now}"
    info_size = 5  # pt
    c.setFont(FOOTER_FONT_NAME, info_size)
    info_x = s
    info_y = by1 - info_size * 2
    c.drawString(info_x, info_y, slug_text)

    c.restoreState()


def embed_output_intent(pdf_path, icc_profile_path):
    if pikepdf is None:
        print("WARNING: pikepdf is not installed; skipping OutputIntent embedding.")
        return

    if not os.path.exists(icc_profile_path):
        print(f"WARNING: ICC profile not found at {icc_profile_path}; skipping OutputIntent embedding.")
        return

    with open(icc_profile_path, "rb") as f:
        icc_bytes = f.read()

    with pikepdf.Pdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        icc_stream = pdf.make_stream(icc_bytes)
        icc_stream[pikepdf.Name("/N")] = 4
        icc_stream[pikepdf.Name("/Alternate")] = pikepdf.Name("/DeviceCMYK")

        output_intent = pdf.make_indirect(
            pikepdf.Dictionary(
                {
                    "/Type": pikepdf.Name("/OutputIntent"),
                    "/S": pikepdf.Name("/GTS_PDFX"),
                    "/OutputConditionIdentifier": OUTPUT_INTENT_IDENTIFIER,
                    "/Info": OUTPUT_INTENT_INFO,
                    "/RegistryName": "https://www.color.org",
                    "/DestOutputProfile": icc_stream,
                }
            )
        )

        pdf.Root["/OutputIntents"] = pikepdf.Array([output_intent])

        # Set TrimBox always; set BleedBox only when bleed/crops are enabled.
        for page in pdf.pages:
            page["/TrimBox"] = pikepdf.Array([
                SLUG, SLUG,
                SLUG + PAGE_SIZE[0], SLUG + PAGE_SIZE[1],
            ])
            if ENABLE_BLEED_AND_CROPS:
                page["/BleedBox"] = pikepdf.Array([
                    SLUG - BLEED, SLUG - BLEED,
                    SLUG + PAGE_SIZE[0] + BLEED, SLUG + PAGE_SIZE[1] + BLEED,
                ])

        pdf.save(pdf_path)

    print(f"Embedded OutputIntent profile into PDF: {icc_profile_path}")


def get_pos_color(word):
    """
    Return a color based on the POS tag for the supplied word.
    Falls back to WORD_COLOR if no POS info is available.
    """
    global words_without_pos_list
    
    if not word:
        return POS_COLOR_OTHER
    
    lookup = PASSED_WORDS_POS_LOOKUP
    if not lookup:
        return POS_COLOR_OTHER 
    
    key = str(word).lower()
    pos_value = lookup.get(key)
    if not pos_value:
        print(f"WARNING: No POS tag found for word '{word}'. Defaulting to other color.")
        if word not in words_without_pos_list:
            words_without_pos_list.append(word)
        return POS_COLOR_OTHER 
    
    pos_value = pos_value.upper()
    if pos_value.startswith("NN"):
        print(f"noun: {word}")
        return POS_COLOR_NOUN
    if pos_value.startswith("JJ"):
        print(f"adj: {word}")
        return POS_COLOR_ADJECTIVE
    if pos_value.startswith("VB"):
        print(f"verb: {word}")
        return POS_COLOR_VERB
    print(f"WARNING: No valid POS tag found for word '{word}'. Defaulting to other color.")
    if word not in words_without_pos_list:
        words_without_pos_list.append(word)
    return POS_COLOR_OTHER 
    
def get_document_topic_weights_simple(model, bow_vector, topic_id):
    # this seems deprecated
    if not bow_vector:
        return 0
    
    # Check if it's a list containing 'none' or 'blank'
    if isinstance(bow_vector, list) and len(bow_vector) > 0:
        first_item = bow_vector[0]
        if first_item == "none":
            return "none"
        elif first_item == "blank":
            return "blank"
    
    # bow_vector should be a list of (word_id, count) tuples like [(0, 1), (5, 2)]
    doc_topics = model.get_document_topics(bow_vector, minimum_probability=0)
    # for w in doc_topics:
    #     if w[1] > .1:
    #         print("topic match, weight:", w[1], "topic", w[0])
    
    topic_weight = 0
    for topic, weight in doc_topics:
        if topic == topic_id:
            topic_weight = weight
            break
    
    return topic_weight

def map_values_to_range(input_list):
    numerical_values = []
    result = []
    
    for value in input_list:
        if value == "none":
            result.append("outline")
        elif value == "blank":
            result.append("italic")
        else:
            numerical_values.append(float(value))
            result.append(None)
    
    # Apply log transform to spread out values
    log_values = np.log1p(numerical_values)  # log1p handles zero values safely
    
    min_val = np.min(log_values)
    max_val = np.max(log_values)
    
    if min_val == max_val:
        mapped_values = [1.0] * len(log_values)
    else:
        mapped_values = [(max_val - val) / (max_val - min_val) for val in log_values]
    
    mapped_index = 0
    for i, value in enumerate(input_list):
        if value != "none" and value != "blank":
            result[i] = mapped_values[mapped_index]
            mapped_index += 1
    return result
# ---------- 0) BATCH  -----------------------------------------------------
if BATCH_PROCESS:
    print("Batch processing")
    for file in sorted(os.listdir(INPUT_PATH)):
        if file.endswith(".csv"):
            csv_info = analyze_csv(file, INPUT_PATH, NUM_ROWS)
            CSV_LIST.update({csv_info[0] : csv_info[1]})  # Use topic number (csv_info[0]) as key
#If not batch, add only the csvs selected from the list
elif len(PROCESS_SELECT) > 0:
    print("processing CSVS " + str(PROCESS_SELECT))
    for file in sorted(os.listdir(INPUT_PATH)):
        for i in PROCESS_SELECT:
            if str(i) in file:
                 if file.endswith(".csv"):
                    csv_info = analyze_csv(file, INPUT_PATH, NUM_ROWS)
                    CSV_LIST.update({csv_info[0] : csv_info[1]})  # Use topic number (csv_info[0]) as key
else:
    print("Add numbers to PROCESS_SELECT or turn on batch processing")

#---------- 0.5) Preprocessing and load model  -----------------------------------------------------
def preprocess(text, MY_STOPWORDS):
    result = []
    text = text.lower()
    for token in gensim.utils.simple_preprocess(text):
        if token not in MY_STOPWORDS and len(token) > 3:
            # print("Not stopped:", token)
            result.append(lemmatize_stemming(token))
    return result


stemmer = SnowballStemmer('english')

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# open and read a csv file, and assign each row as an element in a list
def read_csv(file_path):
    with open(file_path, 'r') as file:
        data_list = file.read().splitlines()
    return data_list


#load the model
lda_model_tfidf = gensim.models.LdaModel.load(MODEL_PATH+'model')
lda_dict = corpora.Dictionary.load(MODEL_PATH+'model.id2word')

# ---------- 1) LOAD DATA -----------------------------------------------------
GENDER_LIST = read_csv(os.path.join(STOPWORD_PATH, "stopwords_gender.csv"))
ETH_LIST = read_csv(os.path.join(STOPWORD_PATH, "stopwords_ethnicity.csv"))
AGE_LIST = read_csv(os.path.join(STOPWORD_PATH, "stopwords_age.csv"))                       
SKIP_TOKEN_LIST = read_csv(os.path.join(STOPWORD_PATH, "skip_tokens.csv"))   
MY_STOPWORDS = (GENDER_LIST+ETH_LIST+AGE_LIST+SKIP_TOKEN_LIST)

#set up dictionary
DICT_PATH=os.path.join(MODEL_PATH,"dictionary.dict")
dictionary = corpora.Dictionary.load(MODEL_PATH+'model.id2word')


if os.path.exists(STOPWORD_DF_PATH):
    topic_word_stopword_df = pd.read_csv(STOPWORD_DF_PATH)
    # Add 'Replace' column if it doesn't exist
    if 'Replace' not in topic_word_stopword_df.columns:
        topic_word_stopword_df['Replace'] = ''
    print(f"Loaded existing stopword data from {STOPWORD_DF_PATH}")
else:
    topic_word_stopword_df = pd.DataFrame(columns=['word', 'stopword', 'stopped', 'Replace'])
    print(f"Created new stopword data file at {STOPWORD_DF_PATH}")


def gray_color(word, font_size, position, orientation, random_state=None, **kw):
    """Return an rgb() string based on stopword status and POS tagging."""
    global _word_color_cache
    global passed_words_list
    if MANUAL_PICK:
        # Check cache first to avoid repeated prompts for the same word
        if word in _word_color_cache:
            return _word_color_cache[word]

        chosen_color = None
        global CSV_NUMBER # Access the global CSV_NUMBER for the current topic
        global topic_word_stopword_df # Access the global DataFrame

        # Check if the word for the current topic has already been processed and stored
        # in the topic_word_stopword_df.
        # This prevents re-prompting for words whose color has already been decided
        # and recorded in a previous run or earlier in the current run.
        existing_entry = topic_word_stopword_df[
            (topic_word_stopword_df['word'] == word)
        ]

        if not existing_entry.empty:
            # If an entry exists, determine the color based on the 'stopped' column
            stopped_value = existing_entry.iloc[0]['stopped']
            replace_value = existing_entry.iloc[0].get('Replace', '') if 'Replace' in existing_entry.columns else ''
            
            # Check if stopped is empty (None, NaN, empty string)
            if pd.isna(stopped_value) or stopped_value == '' or stopped_value is None:
                # If stopped is empty, check the Replace column
                if pd.notna(replace_value) and replace_value != '':
                    # If Replace has a value, set color to black
                    chosen_color = "rgb(0,0,0)"
                    _word_color_cache[word] = chosen_color
                    return chosen_color
                # If Replace is also empty, continue to other logic below
            elif stopped_value == False:
                # If 'stopped' is False, it means the user chose Black (not to stop it)
                chosen_color = "rgb(0,0,0)" 
            elif stopped_value == True:
                # If 'stopped' is True, it means the user chose Gray (to stop it)
                chosen_color = "rgb(180,180,180)"
            
            # Only return if we've set a color (i.e., stopped was not empty/None)
            if chosen_color is not None:
                _word_color_cache[word] = chosen_color
                return chosen_color
        # Iterate through stopwords to find the first match based on priority
        # The first match found will trigger the prompt and return, ensuring only one prompt per word.
        for stopword in MY_STOPWORDS:
            
            title = None
            if stopword == word:
                title = f"Word '{word}' is an exact stopword match with '{stopword}'. Choose color:"
            elif stopword in word:
                # This means 'word' contains 'stopword' (e.g., word="people", stopword="peo")
                title = f"Word '{word}' contains stopword '{stopword}'. Choose color:"
            elif word in stopword:
                # This means 'stopword' contains 'word' (e.g., word="people", stopword="massaii people")
                title = f"Word '{word}' contains stopword '{stopword}'. Choose color:"
            
            if title: # A match was found
                options = [
                    "Black (rgb(0,0,0))",
                    "Gray (rgb(180,180,180))" # Using a medium gray for choice
                ]
                selected_option, index = pick(options, title)
                
                if index == 0: # User chose Black
                    print(f"User chose Black for word '{word}'.")
                    chosen_color = f"rgb(0,0,0)"
                    topic_word_stopword_df.loc[len(topic_word_stopword_df)] = {'word': word, 'stopword': stopword, 'stopped': False}
                else: # User chose Gray
                    print(f"User chose Gray for word '{word}'.")
                    chosen_color = f"rgb(180, 180, 180)"
                    topic_word_stopword_df.loc[len(topic_word_stopword_df)] = {'word': word, 'stopword': stopword, 'stopped': True}
                _word_color_cache[word] = chosen_color # Cache the user's choice for this word
                return chosen_color # Return immediately after the first match and user interaction

        # If the loop completes without finding any stopword match
        chosen_color = f"rgb(0, 0, 0)"  # Black for no match (not in stopwords, good!)
        _word_color_cache[word] = chosen_color # Cache this default choice too
        return chosen_color
    else:
        # Non-MANUAL_PICK mode
        existing_entry = topic_word_stopword_df[
            (topic_word_stopword_df['word'] == word)
        ]

        if not existing_entry.empty:
            stopped_value = existing_entry.iloc[0]['stopped']
            if stopped_value is True:
                return STOPWORD_COLOR
            elif stopped_value is False:
                passed_words_list.append(word)
                return get_pos_color(word)

        # print(f'Word {word} cleared')
        passed_words_list.append(word)
        return get_pos_color(word)
            



def get_all_topics_words(lda_model):
    # Get all topics with their word probabilities
    topics = lda_model.print_topics(num_topics=-1, num_words=len(lda_model.id2word))

    # Or get the topic-word matrix
    topic_word_matrix = lda_model.get_topics()
    # print(topic_word_matrix.shape)  # (num_topics, vocab_size)

    # Get word probabilities for a specific topic
    # topic_0_words = lda_model.show_topic(0, topn=len(lda_model.id2word))

    # Get all words for all topics
    all_topics_words = []
    for topic_id in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=len(lda_model.id2word))
        # print(topic_id, topic_words)
        all_topics_words.append(topic_words)
    return all_topics_words


all_topics_words = get_all_topics_words(lda_model_tfidf)
ensure_output_paths()
cmyk_transform = None
if COLOR_MANAGED_CMYK:
    try:
        cmyk_transform = get_cmyk_transform(ICC_PROFILE_PATH)
        print(f"Loaded CMYK transform using ICC profile: {ICC_PROFILE_PATH}")
    except Exception as exc:
        print(f"WARNING: Could not initialize CMYK transform ({exc}). Continuing without CMYK conversion.")
        cmyk_transform = None

# ---------- FIRST PASS: Collect all frequencies for global scaling ----------
print("First pass: Collecting frequencies for global scaling...")
all_freqs_dicts = []
csv_data_list = []  # Store CSV processing data for second pass

for csv in CSV_LIST:
    CSV_NUMBER = csv.split('topic_')[1].split('_counts.csv')[0].zfill(2)  # Normalize to 2-digit zero-padded string
    print("Processing (pass 1): " + CSV_NUMBER)
    this_topics_words = dict(all_topics_words[int(CSV_NUMBER)])
    df = pd.read_csv(INPUT_PATH+csv).dropna(subset=["description", "count"])
    
    # Build frequencies with replacements
    freqs = build_freqs_with_replacements(df, topic_word_stopword_df)
    
    # Print frequency statistics for this CSV
    if freqs:
        counts = list(freqs.values())
        print(f"  Topic {CSV_NUMBER} - Original frequencies: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.2f}, count={len(freqs)} words")
        # Show top 5 words by frequency
        top_words = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Topic {CSV_NUMBER} - Top 5 words: {[(w, c) for w, c in top_words]}")
    else:
        print(f"  Topic {CSV_NUMBER} - WARNING: No frequencies found!")
    
    all_freqs_dicts.append(freqs)
    
    # Store data needed for second pass (including max frequency for ranking)
    max_freq = max(freqs.values()) if freqs else 0
    csv_data_list.append({
        'csv': csv,
        'CSV_NUMBER': CSV_NUMBER,
        'df': df,
        'this_topics_words': this_topics_words,
        'freqs': freqs,
        'max_freq': max_freq  # Store max frequency for ranking
    })

# Compute a single global linear scale across all topics
print("\n" + "="*60)
print("Computing global linear scale across all CSVs...")
print("="*60)
global_scale = compute_global_scale(all_freqs_dicts, use_log_scale=False)
print(f"\nGlobal scale summary (linear):")
print(f"  Min: {global_scale['min']:.4f}")
print(f"  Max: {global_scale['max']:.4f}")
print(f"  Range: {global_scale['max'] - global_scale['min']:.4f}")
print(f"  This scale will be used for all {len(all_freqs_dicts)} topics")
print("="*60 + "\n")

# Assign topics to the configured buckets (highest frequency first)
csv_data_list.sort(key=lambda x: x['max_freq'], reverse=True)
bucket_summaries = []
topic_index = 0
total_topics = len(csv_data_list)

for bucket in SCALE_BUCKETS:
    bucket_count = bucket.get('topic_count', 0)
    if bucket_count <= 0:
        continue

    bucket_topics = csv_data_list[topic_index: topic_index + bucket_count]
    if not bucket_topics:
        break

    actual_count = len(bucket_topics)
    assigned_ids = []
    for idx, csv_topic in enumerate(bucket_topics):
        if actual_count == 1:
            position = 0.0  # Single topic gets max value (output_max)
        else:
            position = idx / (actual_count - 1)

        output_min = bucket['output_min']
        output_max = bucket['output_max']
        # Invert: position 0.0 (highest freq) → output_max, position 1.0 (lowest freq) → output_min
        bucketed_proportion = output_max - position * (output_max - output_min)
        bucketed_proportion = _clamp(bucketed_proportion)

        csv_topic['bucket_name'] = bucket['name']
        csv_topic['bucket_output_min'] = output_min
        csv_topic['bucket_output_max'] = output_max
        csv_topic['bucket_position'] = position
        csv_topic['bucketed_proportion'] = bucketed_proportion

        assigned_ids.append(csv_topic['CSV_NUMBER'])

    bucket_summaries.append((bucket['name'], assigned_ids))
    topic_index += actual_count

if topic_index < total_topics:
    fallback_bucket = SCALE_BUCKETS[-1] if SCALE_BUCKETS else {"name": "fallback", "output_min": 0.0, "output_max": 1.0}
    bucket_topics = csv_data_list[topic_index:]
    actual_count = len(bucket_topics)
    assigned_ids = []
    for idx, csv_topic in enumerate(bucket_topics):
        if actual_count == 1:
            position = 0.0  # Single topic gets max value (output_max)
        else:
            position = idx / (actual_count - 1)

        output_min = fallback_bucket['output_min']
        output_max = fallback_bucket['output_max']
        # Invert: position 0.0 (highest freq) → output_max, position 1.0 (lowest freq) → output_min
        bucketed_proportion = output_max - position * (output_max - output_min)
        bucketed_proportion = _clamp(bucketed_proportion)

        csv_topic['bucket_name'] = fallback_bucket['name'] + "_overflow"
        csv_topic['bucket_output_min'] = output_min
        csv_topic['bucket_output_max'] = output_max
        csv_topic['bucket_position'] = position
        csv_topic['bucketed_proportion'] = bucketed_proportion

        assigned_ids.append(csv_topic['CSV_NUMBER'])

    bucket_summaries.append((fallback_bucket['name'] + " (overflow)", assigned_ids))

print("\n" + "="*60)
print("Bucket assignments:")
for bucket_name, ids in bucket_summaries:
    display_ids = ids if ids else ["None"]
    print(f"  {bucket_name}: {', '.join(display_ids)}")
print("="*60 + "\n")

# ---------- SECOND PASS: Scale frequencies and create word clouds ----------
print("Second pass: Scaling frequencies and creating word clouds...")
for csv_data in csv_data_list:
    CSV_NUMBER = csv_data['CSV_NUMBER']
    print("Processing (pass 2): " + CSV_NUMBER)
    this_topics_words = csv_data['this_topics_words']
    df = csv_data['df']
    freqs = csv_data['freqs']  # Use pre-computed freqs from first pass

    keyword_list = []
    remove_empty_list = []
    unproccessed_word_dict = {}
    for row in df.iterrows():
        #preprocess the text
        unproccessed_word = row[1].iloc[1]
        tokens = preprocess(unproccessed_word, MY_STOPWORDS)
        # print("token:", tokens)
        keyword_list.append(tokens)
        
        # Check if tokens is empty
        if not tokens:  # This is True when tokens is an empty list
            unproccessed_word_dict["FOOBAR"] = unproccessed_word
        else:
            # print('Unprocessed word dict ', unproccessed_word_dict.get(tokens[0], 'Key not found'))
            unproccessed_word_dict[tokens[0]] = unproccessed_word
    #remove empty lists
    for each in keyword_list:
        if each == []:
            remove_empty_list.append(each)
    for each in remove_empty_list:
        keyword_list.remove(each)

    if CUTOFF and len(keyword_list) > NUM_ROWS:
        keyword_list = keyword_list[:NUM_ROWS]

    #creating bow vector and tokenizing

    # #if the bow vector is empty (only happens if words are in stopword list), replace with "none" for outlining text later
    # bow_vector = []
    # for each in keyword_list:
    #     bow_vector.append(dictionary.doc2bow(each))
    # for i in range(len(bow_vector)):
    #     if len(bow_vector[i]) == 0:
    #         bow_vector[i] = ["none"]
    #     elif len(bow_vector[i]) > 1:
    #         bow_vector[i] = [bow_vector[i][0]]


    sorted_topics = []

    # #get weights in relation to specific topic currently being used
    # for i in range(len(bow_vector)):         
    #     weights = get_document_topic_weights_simple(lda_model_tfidf, bow_vector[i], int(CSV_NUMBER))  
    #     sorted_topics.append(weights)


    for i in range(len(sorted_topics)):
        if sorted_topics[i]== 0.015625:
            sorted_topics[i] = "blank"



    # Scale frequencies using the global linear scale (with optional bucket mapping)
    print(f"  Topic {CSV_NUMBER} - Scaling frequencies...")
    print(f"  Topic {CSV_NUMBER} - Original frequencies: {len(freqs)} words")
    if freqs:
        orig_counts = list(freqs.values())
        print(f"  Topic {CSV_NUMBER} - Original stats: min={min(orig_counts)}, max={max(orig_counts)}, mean={np.mean(orig_counts):.2f}")
    
    if freqs:
        local_max = max(freqs.values())
        min_val = global_scale['min']
        max_val = global_scale['max']
        scale_range = max_val - min_val
        if scale_range > 0:
            raw_proportion = (local_max - min_val) / scale_range
        else:
            raw_proportion = 1.0
        
        raw_proportion = _clamp(raw_proportion)
        bucketed_proportion = csv_data.get('bucketed_proportion', raw_proportion)
        bucket_name = csv_data.get('bucket_name', 'unassigned')
        bucket_position = csv_data.get('bucket_position')
        
        topic_max_font_size = int(FONT_MIN + bucketed_proportion * (FONT_MAX - FONT_MIN))
        
        # Explicitly clamp to ensure it never goes below FONT_MIN (safety check)
        topic_max_font_size = max(FONT_MIN, topic_max_font_size)
        
        # Ensure max_font_size is always greater than min_font_size
        # If they're equal, WordCloud will have no size variation
        if topic_max_font_size <= FONT_MIN:
            topic_max_font_size = FONT_MIN + 1
            print(f"  Topic {CSV_NUMBER} - WARNING: Adjusted max font size to {topic_max_font_size} (was <= FONT_MIN)")
        
        # Warn if max_font_size is very close to FONT_MIN (might indicate scaling issues)
        if topic_max_font_size <= FONT_MIN + 5:
            print(f"  Topic {CSV_NUMBER} - WARNING: Max font size ({topic_max_font_size}) is very close to minimum ({FONT_MIN}) - words will appear very small")
        
        position_display = f"{bucket_position:.2f}" if bucket_position is not None else "n/a"
        print(f"  Topic {CSV_NUMBER} - Max frequency: {local_max}, Bucket: {bucket_name}, Bucket position: {position_display}, Raw proportion: {raw_proportion:.4f}, Bucketed proportion: {bucketed_proportion:.4f}, Max font size: {topic_max_font_size} (min: {FONT_MIN}, max: {FONT_MAX})")
        global_proportion = bucketed_proportion
    else:
        topic_max_font_size = FONT_MAX
        global_proportion = 1.0
        print(f"  Topic {CSV_NUMBER} - WARNING: No frequencies, using default max font size")
    
    # Store original freqs for reference
    original_freqs = freqs.copy()
    
    # Use original frequencies (not pre-scaled) - WordCloud will normalize internally
    # The max_font_size setting will ensure proper global scaling
    wordcloud_freqs = freqs
    
    # Print comparison
    if wordcloud_freqs:
        freq_vals = list(wordcloud_freqs.values())
        print(f"  Topic {CSV_NUMBER} - Frequency stats: min={min(freq_vals)}, max={max(freq_vals)}, mean={np.mean(freq_vals):.2f}")
        # Show top 5 words
        top_words = sorted(wordcloud_freqs.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Topic {CSV_NUMBER} - Top 5 words: {[(w, c) for w, c in top_words]}")
    
    # print("freqs", freqs)
   #relations = dict(zip(df['description'], map_values_to_range(sorted_topics)))
    relations = dict(zip(df['description'], sorted_topics))

    # ---------- 3) BUILD WORD CLOUD ---------------------------------------------
    print(f"  Topic {CSV_NUMBER} - Generating word cloud with {len(wordcloud_freqs)} words (max font: {topic_max_font_size})...")
    wc = (
        WordCloud(width=WC_WIDTH,
                height=WC_HEIGHT,
                background_color=BACKGROUND_COLOR,
                prefer_horizontal=1.0,
                min_font_size=FONT_MIN,
                max_font_size=topic_max_font_size,  # Set max font size based on global proportion
                color_func=gray_color, ##Interpret as colormap rather than single color, use LLM to process in between steps
                font_path=FONT_FILE)
        .generate_from_frequencies(wordcloud_freqs)
    )

    # Save both a permanent PNG at native size and a temp PNG for PDF embedding.
    topic_png_path = os.path.join(OUTPUT_PATH, f"topic_{CSV_NUMBER}.png")
    wc.to_file(topic_png_path)
    tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    wc.to_file(tmp_png.name)
    cmyk_jpg_path = None
    cmyk_image_size = None
    if cmyk_transform is not None:
        cmyk_jpg_path, cmyk_image_size = save_cmyk_jpeg_from_wordcloud(
            wc,
            CSV_NUMBER,
            cmyk_transform,
            ICC_PROFILE_PATH,
        )
        print(f"  Topic {CSV_NUMBER} - CMYK JPEG saved to {cmyk_jpg_path}")
    print(f"  Topic {CSV_NUMBER} - Word cloud generated and saved to {topic_png_path}")
    print(f"  Topic {CSV_NUMBER} - Temporary PDF source saved to {tmp_png.name}")

    # Store the word cloud data for later PDF creation
    PDF_DATA[CSV_NUMBER] = {
        'freqs': original_freqs,  # Store original frequencies
        'max_font_size': topic_max_font_size,  # Store the max font size used
        'global_proportion': global_proportion,  # Store the global proportion
        'relations': relations,
        'topic_png': topic_png_path,
        'cmyk_jpg': cmyk_jpg_path,
        'cmyk_size': cmyk_image_size,
        'tmp_png': tmp_png.name,
        'wc': wc,
        'keyword_list': keyword_list,
        'unprocessed_word_dict': unproccessed_word_dict,
        'this_topics_words': this_topics_words
    }

    # Toggle side for next iteration
    if SIDE == "left":
        SIDE = "right"
    else:
        SIDE = "left"

# ---------- 4) CREATE FINAL PDF WITH ALTERNATING PAGES -------------------------
print("Creating final PDF with alternating pages...")

# Register fonts
pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_FILE))
pdfmetrics.registerFont(TTFont(FOOTER_FONT_NAME, FOOTER_FONT_FILE))

# Create the final PDF
final_pdf_path = OUT_PDF + '.pdf'
c = canvas.Canvas(final_pdf_path, pagesize=CANVAS_SIZE)

# Load footer data from CSV
footer_lookup = {}
if os.path.exists(FOOTER_FILE):
    try:
        footer_df = pd.read_csv(FOOTER_FILE)
        # Create lookup dictionary: topic_id (as string) -> {topic_name, topic_fullname}
        for _, row in footer_df.iterrows():
            topic_id = str(row['topic_id']).zfill(2)  # Normalize to 2-digit zero-padded string
            # topic_id = str(row['topic_id'])
            footer_lookup[topic_id] = {
                'topic_name': str(row['topic name']),
                'topic_fullname': str(row['topic fullname'])
            }
        print(f"Loaded footer data for {len(footer_lookup)} topics from {FOOTER_FILE}")
    except Exception as e:
        print(f"Warning: Could not load footer file {FOOTER_FILE}: {e}")
        footer_lookup = {}
else:
    print(f"Warning: Footer file not found at {FOOTER_FILE}")

#export stopword data to csv
stopword_csv_path = os.path.join(OUTPUT_PATH, "topic_word_stopword.csv")
if not os.path.exists(stopword_csv_path):
    topic_word_stopword_df.to_csv(stopword_csv_path, index=False)

#export passed words to csv (remove duplicates first)
passed_words_df = pd.DataFrame({'word': passed_words_list})
passed_words_df = passed_words_df.drop_duplicates()
passed_words_df = passed_words_df.sort_values(by='word')
passed_words_df.to_csv(os.path.join(OUTPUT_PATH, "passed words.csv"), index=False)

#export words without POS tags to csv (remove duplicates first)
words_without_pos_df = pd.DataFrame({'word': words_without_pos_list})
words_without_pos_df = words_without_pos_df.drop_duplicates()
words_without_pos_df = words_without_pos_df.sort_values(by='word')
words_without_pos_df.to_csv(os.path.join(OUTPUT_PATH, "passed_words_without_pos.csv"), index=False)

# Process each CSV in order, alternating left/right
current_side = "right"
page_number = 1
for csv in CSV_LIST:
    CSV_NUMBER = csv.split('topic_')[1].split('_counts.csv')[0].zfill(2)  # Normalize to 2-digit zero-padded string
    
    if CSV_NUMBER not in PDF_DATA:
        continue
        
    data = PDF_DATA[CSV_NUMBER]
    tmp_png = data['tmp_png']
    wc = data['wc']
    cmyk_jpg = data.get('cmyk_jpg')
    cmyk_size = data.get('cmyk_size')
    
    # Calculate margins based on current side
    if current_side == "left":
        left_margin = OUTER_MARGIN
        right_margin = INNER_MARGIN
    else:
        left_margin = INNER_MARGIN
        right_margin = OUTER_MARGIN
    
    # Calculate available space for word cloud
    available_width = PAGE_SIZE[0] - left_margin - right_margin
    available_height = PAGE_SIZE[1] - TOP_MARGIN - BOTTOM_MARGIN
    
    # Load image
    if cmyk_jpg:
        img = ImageReader(cmyk_jpg)
        img_width, img_height = cmyk_size
    else:
        img = ImageReader(tmp_png)
        img_width, img_height = wc.to_image().size
    
    # Calculate scale to fit within available space
    scale_x = available_width / img_width
    scale_y = available_height / img_height
    scale = min(scale_x, scale_y)# * 0.9  # 90% of max size for some padding
    
    # Calculate final dimensions
    draw_w = img_width * scale
    draw_h = img_height * scale
    
    # Calculate position (centered within trim area, offset by slug for bleed/marks)
    x = SLUG + left_margin + (available_width - draw_w) / 2
    y = SLUG + BOTTOM_MARGIN + (available_height - draw_h) / 2

    # Draw the word cloud
    c.drawImage(img, x, y, width=draw_w, height=draw_h, mask="auto")
    
    # Add footer
    c.saveState()
    c.setFont(FOOTER_FONT_NAME, FOOTER_FONT_SIZE)
    print("CSV_NUMBER", CSV_NUMBER)
    # Get footer data from CSV lookup
    footer_data = footer_lookup.get(CSV_NUMBER, {})
    print("footer_data", footer_data)
    if footer_data:
        topic_fullname = footer_data.get('topic_fullname', 'No topic name available')
        if CSV_NUMBER[0] == '0':
            CSV_NUMBER = CSV_NUMBER[1:]
        footer_text = f"Topic {CSV_NUMBER} ({topic_fullname}, etc.)"
    else:
        footer_text = f"Topic {CSV_NUMBER} (No topic data available)"
    
    # Choose footer alignment based on even/odd page (left/right side)
    FOOTER_INSET = int(0.125 * 72)  # 1/8 inch in points

    if current_side == "left":
        footer_x = SLUG + left_margin + FOOTER_INSET
    else:
        footer_x = SLUG + PAGE_SIZE[0] - right_margin - c.stringWidth(footer_text, FOOTER_FONT_NAME, FOOTER_FONT_SIZE) - FOOTER_INSET

    # Footer text vertically centered in bottom margin (using font ascent/descent)
    ascent_pt = pdfmetrics.getAscent(FOOTER_FONT_NAME) * FOOTER_FONT_SIZE / 1000
    descent_pt = pdfmetrics.getDescent(FOOTER_FONT_NAME) * FOOTER_FONT_SIZE / 1000  # negative
    text_center_offset = (ascent_pt + descent_pt) / 2
    footer_y = SLUG + BOTTOM_MARGIN / 2 - text_center_offset + FOOTER_NUDGE
    c.drawString(footer_x, footer_y, footer_text)
    c.restoreState()
    
    if ENABLE_BLEED_AND_CROPS:
        draw_print_marks(c, footer_text, page_number)
    c.showPage()

    # Toggle side for next page
    current_side = "right" if current_side == "left" else "left"
    page_number += 1

c.save()
embed_output_intent(final_pdf_path, ICC_PROFILE_PATH)
print(f"✅ Final word-cloud PDF created → {final_pdf_path}")

# Clean up temporary files
for csv in CSV_LIST:
    CSV_NUMBER = csv.split('topic_')[1].split('_counts.csv')[0].zfill(2)  # Normalize to 2-digit zero-padded string
    if CSV_NUMBER in PDF_DATA:
        try:
            os.unlink(PDF_DATA[CSV_NUMBER]['tmp_png'])
        except:
            pass

