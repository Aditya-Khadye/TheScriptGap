"""
===============================================================================
Script Diversity Index — ViT-B/16 Embedding Pipeline (100-Glyph Edition)
===============================================================================
Project:  TRC / Monotype — Identifying Underserved Scripts
Author:   Aditya (UCF MIT2 Lab)
Purpose:  Measure the visual diversity of fonts within each writing system
          using a pretrained Vision Transformer as a feature extractor.

Changes from 10-glyph version:
    - 100 reference characters per script (was 10)
    - Characters organized into tiers: base consonants, vowels, complex
      forms (conjuncts/ligatures), numerals, signs/marks
    - More representative sampling of each script's visual range
    - Better signal for the ViT to distinguish font styles

    Why 100? At 10 glyphs, we only sampled simple base consonants. But
    font designers make their most distinctive choices on complex forms —
    conjuncts in Devanagari, ligatures in Arabic, vowel combinations in
    Tamil. Missing these means missing the diversity signal. 100 glyphs
    covers base forms, complex forms, numerals, and marks — giving the
    ViT a comprehensive view of each font's design decisions.

Architecture: ViT-B/16 (Vision Transformer, Base, 16×16 patches)
    - Frozen pretrained weights (ImageNet1K_V1)
    - 768-dimensional embeddings from [CLS] token
    - L2-normalized for cosine distance computation

Data Source: Google Fonts (github.com/google/fonts)
Usage:
    1. Ensure ./fonts/ exists (git clone --depth 1 https://github.com/google/fonts)
    2. Run: python script_diversity_vit_100.py
    3. Output: vit_outputs_100/diversity_index_results.csv
===============================================================================
"""

import os
import sys
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GOOGLE_FONTS_DIR = Path("./fonts")
OUTPUT_DIR = Path("./vit_outputs_100")

FONT_SIMILARITY_DIR = Path("./vit_outputs_100/font_similarity_pairs")
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"

CANVAS_SIZE = 224
FONT_RENDER_SIZE = 140  # slightly smaller to fit complex glyphs
MIN_CODEPOINT_COVERAGE = 10
MAX_FONTS_PER_SCRIPT = 100

# ---------------------------------------------------------------------------
# 100 Reference Characters Per Script
# ---------------------------------------------------------------------------
# Organized into tiers for analysis:
#   Tier 1: Base consonants (structural backbone of the script)
#   Tier 2: Vowels and vowel signs (diacritics, marks)
#   Tier 3: Complex forms (conjuncts, ligatures, combinations)
#   Tier 4: Numerals (often styled differently from letters)
#   Tier 5: Punctuation and signs (supplementary)
#
# Selection criteria:
#   - High frequency in natural text (ensures most fonts include them)
#   - Structural variety (simple strokes to complex compound forms)
#   - Covers the full visual range of the script
# ---------------------------------------------------------------------------

REFERENCE_CHARS = {
    "Devanagari": {
        "base_consonants": [
            "\u0915",  # क ka
            "\u0916",  # ख kha
            "\u0917",  # ग ga
            "\u0918",  # घ gha
            "\u0919",  # ङ nga
            "\u091A",  # च ca
            "\u091B",  # छ cha
            "\u091C",  # ज ja
            "\u091D",  # झ jha
            "\u091E",  # ञ nya
            "\u091F",  # ट tta
            "\u0920",  # ठ ttha
            "\u0921",  # ड dda
            "\u0922",  # ढ ddha
            "\u0923",  # ण nna
            "\u0924",  # त ta
            "\u0925",  # थ tha
            "\u0926",  # द da
            "\u0927",  # ध dha
            "\u0928",  # न na
            "\u092A",  # प pa
            "\u092B",  # फ pha
            "\u092C",  # ब ba
            "\u092D",  # भ bha
            "\u092E",  # म ma
            "\u092F",  # य ya
            "\u0930",  # र ra
            "\u0932",  # ल la
            "\u0935",  # व va
            "\u0936",  # श sha
            "\u0937",  # ष ssa
            "\u0938",  # स sa
            "\u0939",  # ह ha
        ],
        "vowels": [
            "\u0905",  # अ a
            "\u0906",  # आ aa
            "\u0907",  # इ i
            "\u0908",  # ई ii
            "\u0909",  # उ u
            "\u090A",  # ऊ uu
            "\u090B",  # ऋ r
            "\u090F",  # ए e
            "\u0910",  # ऐ ai
            "\u0913",  # ओ o
            "\u0914",  # औ au
        ],
        "vowel_signs": [
            "\u093E",  # ा aa
            "\u093F",  # ि i
            "\u0940",  # ी ii
            "\u0941",  # ु u
            "\u0942",  # ू uu
            "\u0943",  # ृ r
            "\u0947",  # े e
            "\u0948",  # ै ai
            "\u094B",  # ो o
            "\u094C",  # ौ au
            "\u094D",  # ् virama
        ],
        "complex_forms": [
            "\u0915\u094D\u0937",  # क्ष ksha
            "\u0924\u094D\u0930",  # त्र tra
            "\u091C\u094D\u091E",  # ज्ञ jnya
            "\u0936\u094D\u0930",  # श्र shra
            "\u0915\u094D\u0924",  # क्त kta
            "\u0926\u094D\u0927",  # द्ध ddha
            "\u0928\u094D\u0924",  # न्त nta
            "\u0938\u094D\u0924",  # स्त sta
            "\u0915\u094D\u0930",  # क्र kra
            "\u092A\u094D\u0930",  # प्र pra
            "\u0926\u094D\u0935",  # द्व dva
            "\u0939\u094D\u092E",  # ह्म hma
            "\u0915\u093F",        # कि ki
            "\u0915\u0940",        # की kii
            "\u0915\u0941",        # कु ku
            "\u0915\u0942",        # कू kuu
            "\u0915\u0947",        # के ke
            "\u0915\u0948",        # कै kai
            "\u0915\u094B",        # को ko
            "\u0915\u094C",        # कौ kau
        ],
        "numerals": [
            "\u0966",  # ० 0
            "\u0967",  # १ 1
            "\u0968",  # २ 2
            "\u0969",  # ३ 3
            "\u096A",  # ४ 4
            "\u096B",  # ५ 5
            "\u096C",  # ६ 6
            "\u096D",  # ७ 7
            "\u096E",  # ८ 8
            "\u096F",  # ९ 9
        ],
        "signs": [
            "\u0950",  # ॐ om
            "\u0964",  # । danda
            "\u0965",  # ॥ double danda
            "\u0902",  # ं anusvara
            "\u0903",  # ः visarga
            "\u0901",  # ँ chandrabindu
            "\u093C",  # ़ nukta
            "\u0970",  # ॰ abbreviation
            "\u0971",  # ॱ high spacing dot
            "\u090D",  # ऍ candra e
            "\u0911",  # ऑ candra o
            "\u0945",  # ॅ candra e sign
            "\u0949",  # ॉ candra o sign
            "\u0929",  # ऩ nnna
            "\u0931",  # ऱ rra
            "\u0934",  # ऴ llla
        ],
    },

    "Arabic": {
        "base_consonants": [
            "\u0627",  # ا alef
            "\u0628",  # ب ba
            "\u062A",  # ت ta
            "\u062B",  # ث tha
            "\u062C",  # ج jim
            "\u062D",  # ح ha
            "\u062E",  # خ kha
            "\u062F",  # د dal
            "\u0630",  # ذ dhal
            "\u0631",  # ر ra
            "\u0632",  # ز zay
            "\u0633",  # س sin
            "\u0634",  # ش shin
            "\u0635",  # ص sad
            "\u0636",  # ض dad
            "\u0637",  # ط ta
            "\u0638",  # ظ za
            "\u0639",  # ع ain
            "\u063A",  # غ ghain
            "\u0641",  # ف fa
            "\u0642",  # ق qaf
            "\u0643",  # ك kaf
            "\u0644",  # ل lam
            "\u0645",  # م mim
            "\u0646",  # ن nun
            "\u0647",  # ه ha
            "\u0648",  # و waw
            "\u064A",  # ي ya
        ],
        "vowels_marks": [
            "\u064B",  # ً fathatan
            "\u064C",  # ٌ dammatan
            "\u064D",  # ٍ kasratan
            "\u064E",  # َ fatha
            "\u064F",  # ُ damma
            "\u0650",  # ِ kasra
            "\u0651",  # ّ shadda
            "\u0652",  # ْ sukun
            "\u0670",  # ٰ superscript alef
            "\u0671",  # ٱ wasla
        ],
        "complex_forms": [
            "\u0644\u0627",        # لا lam-alef ligature
            "\u0644\u0623",        # لأ lam-alef hamza above
            "\u0644\u0625",        # لإ lam-alef hamza below
            "\u0644\u0622",        # لآ lam-alef madda
            "\u0628\u0633\u0645",  # بسم bismillah start
            "\u0623",  # أ alef hamza above
            "\u0624",  # ؤ waw hamza
            "\u0625",  # إ alef hamza below
            "\u0626",  # ئ ya hamza
            "\u0622",  # آ alef madda
            "\u0629",  # ة ta marbuta
            "\u0649",  # ى alef maksura
            "\u066E",  # ٮ dotless ba
            "\u066F",  # ٯ dotless qaf
            "\u0679",  # ٹ tteh (Urdu)
            "\u067E",  # پ peh (Urdu/Farsi)
            "\u0686",  # چ tcheh (Urdu/Farsi)
            "\u0698",  # ژ jeh (Farsi)
            "\u06A9",  # ک keheh (Farsi)
            "\u06AF",  # گ gaf (Farsi/Urdu)
            "\u06CC",  # ی farsi yeh
            "\u06D2",  # ے yeh barree (Urdu)
        ],
        "numerals": [
            "\u0660",  # ٠ 0
            "\u0661",  # ١ 1
            "\u0662",  # ٢ 2
            "\u0663",  # ٣ 3
            "\u0664",  # ٤ 4
            "\u0665",  # ٥ 5
            "\u0666",  # ٦ 6
            "\u0667",  # ٧ 7
            "\u0668",  # ٨ 8
            "\u0669",  # ٩ 9
            "\u06F0",  # ۰ extended 0
            "\u06F1",  # ۱ extended 1
            "\u06F2",  # ۲ extended 2
            "\u06F3",  # ۳ extended 3
            "\u06F4",  # ۴ extended 4
            "\u06F5",  # ۵ extended 5
            "\u06F6",  # ۶ extended 6
            "\u06F7",  # ۷ extended 7
            "\u06F8",  # ۸ extended 8
            "\u06F9",  # ۹ extended 9
        ],
        "signs": [
            "\u060C",  # ، comma
            "\u061B",  # ؛ semicolon
            "\u061F",  # ؟ question
            "\u0640",  # ـ tatweel
            "\u066A",  # ٪ percent
            "\u066B",  # ٫ decimal separator
            "\u066C",  # ٬ thousands separator
            "\u066D",  # ٭ five pointed star
            "\u06D4",  # ۔ full stop
            "\u0600",  # ؀ number sign
        ],
    },

    "Bengali": {
        "base_consonants": [
            "\u0995",  # ক ka
            "\u0996",  # খ kha
            "\u0997",  # গ ga
            "\u0998",  # ঘ gha
            "\u0999",  # ঙ nga
            "\u099A",  # চ ca
            "\u099B",  # ছ cha
            "\u099C",  # জ ja
            "\u099D",  # ঝ jha
            "\u099E",  # ঞ nya
            "\u099F",  # ট tta
            "\u09A0",  # ঠ ttha
            "\u09A1",  # ড dda
            "\u09A2",  # ঢ ddha
            "\u09A3",  # ণ nna
            "\u09A4",  # ত ta
            "\u09A5",  # থ tha
            "\u09A6",  # দ da
            "\u09A7",  # ধ dha
            "\u09A8",  # ন na
            "\u09AA",  # প pa
            "\u09AB",  # ফ pha
            "\u09AC",  # ব ba
            "\u09AD",  # ভ bha
            "\u09AE",  # ম ma
            "\u09AF",  # য ya
            "\u09B0",  # র ra
            "\u09B2",  # ল la
            "\u09B6",  # শ sha
            "\u09B7",  # ষ ssa
            "\u09B8",  # স sa
            "\u09B9",  # হ ha
        ],
        "vowels": [
            "\u0985",  # অ a
            "\u0986",  # আ aa
            "\u0987",  # ই i
            "\u0988",  # ঈ ii
            "\u0989",  # উ u
            "\u098A",  # ঊ uu
            "\u098B",  # ঋ r
            "\u098F",  # এ e
            "\u0990",  # ঐ ai
            "\u0993",  # ও o
            "\u0994",  # ঔ au
        ],
        "vowel_signs": [
            "\u09BE",  # া aa
            "\u09BF",  # ি i
            "\u09C0",  # ী ii
            "\u09C1",  # ু u
            "\u09C2",  # ূ uu
            "\u09C3",  # ৃ r
            "\u09C7",  # ে e
            "\u09C8",  # ৈ ai
            "\u09CB",  # ো o
            "\u09CC",  # ৌ au
            "\u09CD",  # ্ virama
        ],
        "complex_forms": [
            "\u0995\u09CD\u09B7",  # ক্ষ ksha
            "\u0995\u09CD\u09A4",  # ক্ত kta
            "\u09A8\u09CD\u09A4",  # ন্ত nta
            "\u09B8\u09CD\u09A4",  # স্ত sta
            "\u09A6\u09CD\u09A7",  # দ্ধ ddha
            "\u0995\u09CD\u09B0",  # ক্র kra
            "\u09AA\u09CD\u09B0",  # প্র pra
            "\u09A4\u09CD\u09B0",  # ত্র tra
            "\u0995\u09BF",        # কি ki
            "\u0995\u09C0",        # কী kii
            "\u0995\u09C1",        # কু ku
            "\u0995\u09C7",        # কে ke
            "\u0995\u09CB",        # কো ko
            "\u0995\u09CC",        # কৌ kau
            "\u09DC",  # ড় rra
            "\u09DD",  # ঢ় rrha
            "\u09DF",  # য় yya
        ],
        "numerals": [
            "\u09E6",  # ০ 0
            "\u09E7",  # ১ 1
            "\u09E8",  # ২ 2
            "\u09E9",  # ৩ 3
            "\u09EA",  # ৪ 4
            "\u09EB",  # ৫ 5
            "\u09EC",  # ৬ 6
            "\u09ED",  # ৭ 7
            "\u09EE",  # ৮ 8
            "\u09EF",  # ৯ 9
        ],
        "signs": [
            "\u0982",  # ং anusvara
            "\u0983",  # ঃ visarga
            "\u0981",  # ঁ chandrabindu
            "\u09F2",  # ৲ rupee mark
            "\u09F3",  # ৳ rupee sign
            "\u09F7",  # ৷ currency numerator
            "\u09FA",  # ৺ isshar
            "\u09F8",  # ৸ currency numerator
            "\u09F9",  # ৹ currency denominator
            "\u09F0",  # ৰ assamese ra
        ],
    },

    "Tamil": {
        "base_consonants": [
            "\u0B95",  # க ka
            "\u0B99",  # ங nga
            "\u0B9A",  # ச ca
            "\u0B9E",  # ஞ nya
            "\u0B9F",  # ட tta
            "\u0BA3",  # ண nna
            "\u0BA4",  # த ta
            "\u0BA8",  # ந na
            "\u0BAA",  # ப pa
            "\u0BAE",  # ம ma
            "\u0BAF",  # ய ya
            "\u0BB0",  # ர ra
            "\u0BB2",  # ல la
            "\u0BB5",  # வ va
            "\u0BB4",  # ழ llla
            "\u0BB3",  # ள lla
            "\u0BB1",  # ற rra
            "\u0BA9",  # ன nnna
            "\u0B9C",  # ஜ ja (grantha)
            "\u0BB6",  # ஶ sha (grantha)
            "\u0BB7",  # ஷ ssa (grantha)
            "\u0BB8",  # ஸ sa (grantha)
            "\u0BB9",  # ஹ ha (grantha)
        ],
        "vowels": [
            "\u0B85",  # அ a
            "\u0B86",  # ஆ aa
            "\u0B87",  # இ i
            "\u0B88",  # ஈ ii
            "\u0B89",  # உ u
            "\u0B8A",  # ஊ uu
            "\u0B8E",  # எ e
            "\u0B8F",  # ஏ ee
            "\u0B90",  # ஐ ai
            "\u0B92",  # ஒ o
            "\u0B93",  # ஓ oo
            "\u0B94",  # ஔ au
        ],
        "vowel_signs": [
            "\u0BBE",  # ா aa
            "\u0BBF",  # ி i
            "\u0BC0",  # ீ ii
            "\u0BC1",  # ு u
            "\u0BC2",  # ூ uu
            "\u0BC6",  # ெ e
            "\u0BC7",  # ே ee
            "\u0BC8",  # ை ai
            "\u0BCA",  # ொ o
            "\u0BCB",  # ோ oo
            "\u0BCC",  # ௌ au
            "\u0BCD",  # ் virama
        ],
        "complex_forms": [
            "\u0B95\u0BBE",  # கா kaa
            "\u0B95\u0BBF",  # கி ki
            "\u0B95\u0BC0",  # கீ kii
            "\u0B95\u0BC1",  # கு ku
            "\u0B95\u0BC2",  # கூ kuu
            "\u0B95\u0BC6",  # கெ ke
            "\u0B95\u0BC7",  # கே kee
            "\u0B95\u0BC8",  # கை kai
            "\u0B95\u0BCA",  # கொ ko
            "\u0B95\u0BCB",  # கோ koo
            "\u0B95\u0BCC",  # கௌ kau
            "\u0BA4\u0BBE",  # தா taa
            "\u0BAA\u0BBF",  # பி pi
            "\u0BAE\u0BC0",  # மீ mii
            "\u0BB0\u0BC1",  # ரு ru
            "\u0BB2\u0BC6",  # லெ le
            "\u0BA8\u0BC7",  # நே nee
            "\u0BB5\u0BC8",  # வை vai
            "\u0B9F\u0BCD",  # ட் halant tta
            "\u0BA4\u0BCD",  # த் halant ta
            "\u0B95\u0BCD\u0BB7",  # க்ஷ ksha
            "\u0BB8\u0BCD\u0BB0\u0BC0",  # ஸ்ரீ shrii
        ],
        "numerals": [
            "\u0BE6",  # ௦ 0
            "\u0BE7",  # ௧ 1
            "\u0BE8",  # ௨ 2
            "\u0BE9",  # ௩ 3
            "\u0BEA",  # ௪ 4
            "\u0BEB",  # ௫ 5
            "\u0BEC",  # ௬ 6
            "\u0BED",  # ௭ 7
            "\u0BEE",  # ௮ 8
            "\u0BEF",  # ௯ 9
            "\u0BF0",  # ௰ ten
            "\u0BF1",  # ௱ hundred
            "\u0BF2",  # ௲ thousand
        ],
        "signs": [
            "\u0B82",  # ஂ anusvara
            "\u0B83",  # ஃ visarga (aytham)
            "\u0BD0",  # ௐ om
            "\u0BF3",  # ௳ day sign
            "\u0BF4",  # ௴ month sign
            "\u0BF5",  # ௵ year sign
            "\u0BF6",  # ௶ debit
            "\u0BF7",  # ௷ credit
            "\u0BF8",  # ௸ as above
            "\u0BF9",  # ௹ rupee
            "\u0BFA",  # ௺ number sign
        ],
    },

    "Telugu": {
        "base_consonants": [
            "\u0C15",  # క ka
            "\u0C16",  # ఖ kha
            "\u0C17",  # గ ga
            "\u0C18",  # ఘ gha
            "\u0C19",  # ఙ nga
            "\u0C1A",  # చ ca
            "\u0C1B",  # ఛ cha
            "\u0C1C",  # జ ja
            "\u0C1D",  # ఝ jha
            "\u0C1E",  # ఞ nya
            "\u0C1F",  # ట tta
            "\u0C20",  # ఠ ttha
            "\u0C21",  # డ dda
            "\u0C22",  # ఢ ddha
            "\u0C23",  # ణ nna
            "\u0C24",  # త ta
            "\u0C25",  # థ tha
            "\u0C26",  # ద da
            "\u0C27",  # ధ dha
            "\u0C28",  # న na
            "\u0C2A",  # ప pa
            "\u0C2B",  # ఫ pha
            "\u0C2C",  # బ ba
            "\u0C2D",  # భ bha
            "\u0C2E",  # మ ma
            "\u0C2F",  # య ya
            "\u0C30",  # ర ra
            "\u0C31",  # ఱ rra
            "\u0C32",  # ల la
            "\u0C33",  # ళ lla
            "\u0C35",  # వ va
            "\u0C36",  # శ sha
            "\u0C37",  # ష ssa
            "\u0C38",  # స sa
            "\u0C39",  # హ ha
        ],
        "vowels": [
            "\u0C05",  # అ a
            "\u0C06",  # ఆ aa
            "\u0C07",  # ఇ i
            "\u0C08",  # ఈ ii
            "\u0C09",  # ఉ u
            "\u0C0A",  # ఊ uu
            "\u0C0B",  # ఋ r
            "\u0C0E",  # ఎ e
            "\u0C0F",  # ఏ ee
            "\u0C10",  # ఐ ai
            "\u0C12",  # ఒ o
            "\u0C13",  # ఓ oo
            "\u0C14",  # ఔ au
        ],
        "vowel_signs": [
            "\u0C3E",  # ా aa
            "\u0C3F",  # ి i
            "\u0C40",  # ీ ii
            "\u0C41",  # ు u
            "\u0C42",  # ూ uu
            "\u0C43",  # ృ r
            "\u0C46",  # ె e
            "\u0C47",  # ే ee
            "\u0C48",  # ై ai
            "\u0C4A",  # ొ o
            "\u0C4B",  # ో oo
            "\u0C4C",  # ౌ au
            "\u0C4D",  # ్ virama
        ],
        "complex_forms": [
            "\u0C15\u0C3E",  # కా kaa
            "\u0C15\u0C3F",  # కి ki
            "\u0C15\u0C41",  # కు ku
            "\u0C15\u0C47",  # కే kee
            "\u0C15\u0C4B",  # కో koo
            "\u0C24\u0C4D\u0C30",  # త్ర tra
            "\u0C15\u0C4D\u0C37",  # క్ష ksha
            "\u0C15\u0C4D\u0C15",  # క్క kka
            "\u0C24\u0C4D\u0C24",  # త్త tta
            "\u0C28\u0C4D\u0C28",  # న్న nna
            "\u0C2E\u0C4D\u0C2E",  # మ్మ mma
            "\u0C32\u0C4D\u0C32",  # ల్ల lla
            "\u0C38\u0C4D\u0C24",  # స్త sta
            "\u0C30\u0C4D",        # ర్ halant ra
            "\u0C28\u0C4D",        # న్ halant na
        ],
        "numerals": [
            "\u0C66",  # ౦ 0
            "\u0C67",  # ౧ 1
            "\u0C68",  # ౨ 2
            "\u0C69",  # ౩ 3
            "\u0C6A",  # ౪ 4
            "\u0C6B",  # ౫ 5
            "\u0C6C",  # ౬ 6
            "\u0C6D",  # ౭ 7
            "\u0C6E",  # ౮ 8
            "\u0C6F",  # ౯ 9
        ],
        "signs": [
            "\u0C02",  # ం anusvara
            "\u0C03",  # ః visarga
            "\u0C01",  # ఁ chandrabindu
            "\u0C44",  # ౄ rr
        ],
    },

    "Cyrillic": {
        "uppercase": [
            "\u0410",  # А
            "\u0411",  # Б
            "\u0412",  # В
            "\u0413",  # Г
            "\u0414",  # Д
            "\u0415",  # Е
            "\u0416",  # Ж
            "\u0417",  # З
            "\u0418",  # И
            "\u0419",  # Й
            "\u041A",  # К
            "\u041B",  # Л
            "\u041C",  # М
            "\u041D",  # Н
            "\u041E",  # О
            "\u041F",  # П
            "\u0420",  # Р
            "\u0421",  # С
            "\u0422",  # Т
            "\u0423",  # У
            "\u0424",  # Ф
            "\u0425",  # Х
            "\u0426",  # Ц
            "\u0427",  # Ч
            "\u0428",  # Ш
            "\u0429",  # Щ
            "\u042A",  # Ъ
            "\u042B",  # Ы
            "\u042C",  # Ь
            "\u042D",  # Э
            "\u042E",  # Ю
            "\u042F",  # Я
        ],
        "lowercase": [
            "\u0430",  # а
            "\u0431",  # б
            "\u0432",  # в
            "\u0433",  # г
            "\u0434",  # д
            "\u0435",  # е
            "\u0436",  # ж
            "\u0437",  # з
            "\u0438",  # и
            "\u0439",  # й
            "\u043A",  # к
            "\u043B",  # л
            "\u043C",  # м
            "\u043D",  # н
            "\u043E",  # о
            "\u043F",  # п
            "\u0440",  # р
            "\u0441",  # с
            "\u0442",  # т
            "\u0443",  # у
            "\u0444",  # ф
            "\u0445",  # х
            "\u0446",  # ц
            "\u0447",  # ч
            "\u0448",  # ш
            "\u0449",  # щ
            "\u044A",  # ъ
            "\u044B",  # ы
            "\u044C",  # ь
            "\u044D",  # э
            "\u044E",  # ю
            "\u044F",  # я
        ],
        "extended": [
            "\u0401",  # Ё
            "\u0451",  # ё
            "\u0490",  # Ґ Ukrainian
            "\u0491",  # ґ
            "\u0404",  # Є Ukrainian
            "\u0454",  # є
            "\u0406",  # І Ukrainian/Belarusian
            "\u0456",  # і
            "\u0407",  # Ї Ukrainian
            "\u0457",  # ї
            "\u040E",  # Ў Belarusian
            "\u045E",  # ў
            "\u0402",  # Ђ Serbian
            "\u0452",  # ђ
            "\u0409",  # Љ Serbian
            "\u0459",  # љ
            "\u040A",  # Њ Serbian
            "\u045A",  # њ
            "\u040B",  # Ћ Serbian
            "\u045B",  # ћ
            "\u040F",  # Џ Serbian
            "\u045F",  # џ
            "\u0405",  # Ѕ Macedonian
            "\u0455",  # ѕ
            "\u0408",  # Ј Serbian
            "\u0458",  # ј
            "\u040C",  # Ќ Macedonian
            "\u045C",  # ќ
            "\u040D",  # Ѝ Bulgarian
            "\u045D",  # ѝ
            "\u0462",  # Ѣ yat (historical)
            "\u0463",  # ѣ
            "\u046A",  # Ѫ big yus (historical)
            "\u046B",  # ѫ
        ],
        "numerals_punct": [
            "\u2116",  # № numero sign
        ],
    },

    "Han": {
        "high_frequency": [
            "\u7684",  # 的 de (possessive)
            "\u4E00",  # 一 yi (one)
            "\u662F",  # 是 shi (is)
            "\u4E0D",  # 不 bu (not)
            "\u4E86",  # 了 le (completed)
            "\u4EBA",  # 人 ren (person)
            "\u6211",  # 我 wo (I/me)
            "\u5728",  # 在 zai (at/in)
            "\u6709",  # 有 you (have)
            "\u4ED6",  # 他 ta (he)
            "\u8FD9",  # 这 zhe (this)
            "\u4E2D",  # 中 zhong (middle)
            "\u5927",  # 大 da (big)
            "\u6765",  # 来 lai (come)
            "\u4E0A",  # 上 shang (up)
            "\u56FD",  # 国 guo (country)
            "\u4E2A",  # 个 ge (measure word)
            "\u5230",  # 到 dao (arrive)
            "\u8BF4",  # 说 shuo (say)
            "\u4EEC",  # 们 men (plural)
            "\u4E3A",  # 为 wei (for)
            "\u5B50",  # 子 zi (child)
            "\u548C",  # 和 he (and)
            "\u4F60",  # 你 ni (you)
            "\u5730",  # 地 di (earth)
        ],
        "structural_variety": [
            "\u5B57",  # 字 zi (character) — enclosed
            "\u6587",  # 文 wen (writing) — cross strokes
            "\u7528",  # 用 yong (use) — enclosed
            "\u5FC3",  # 心 xin (heart) — radical
            "\u6C34",  # 水 shui (water) — diagonal
            "\u706B",  # 火 huo (fire) — symmetrical
            "\u6728",  # 木 mu (wood) — tree radical
            "\u91D1",  # 金 jin (gold) — complex top
            "\u571F",  # 土 tu (earth) — simple
            "\u5C71",  # 山 shan (mountain) — peaks
            "\u65E5",  # 日 ri (sun) — enclosed
            "\u6708",  # 月 yue (moon) — open
            "\u98A8",  # 風 feng (wind) — complex
            "\u96E8",  # 雨 yu (rain) — top heavy
            "\u96EA",  # 雪 xue (snow) — stacked
            "\u82B1",  # 花 hua (flower) — radical+phonetic
            "\u9B5A",  # 魚 yu (fish) — complex
            "\u9CE5",  # 鳥 niao (bird) — complex
            "\u9F8D",  # 龍 long (dragon) — very complex
            "\u9F9C",  # 龜 gui (turtle) — very complex
            "\u7FBD",  # 羽 yu (feather) — repeated
            "\u7AF9",  # 竹 zhu (bamboo) — repeated
            "\u7CF8",  # 糸 mi (thread) — fine strokes
            "\u8EAB",  # 身 shen (body) — elongated
            "\u9053",  # 道 dao (way) — radical+complex
        ],
        "radicals_components": [
            "\u4EBB",  # 亻 person radical
            "\u53E3",  # 口 mouth
            "\u5973",  # 女 woman
            "\u5B50",  # 子 child
            "\u5C0F",  # 小 small
            "\u5DE5",  # 工 work
            "\u5F13",  # 弓 bow
            "\u624B",  # 手 hand
            "\u65B9",  # 方 square
            "\u6B62",  # 止 stop
            "\u6BDB",  # 毛 hair/fur
            "\u6C14",  # 气 air
            "\u7247",  # 片 slice
            "\u7259",  # 牙 tooth
            "\u7389",  # 玉 jade
            "\u74E6",  # 瓦 tile
            "\u7530",  # 田 field
            "\u76AE",  # 皮 skin
            "\u76EE",  # 目 eye
            "\u77F3",  # 石 stone
            "\u7ACB",  # 立 stand
            "\u8033",  # 耳 ear
            "\u8089",  # 肉 meat
            "\u8A00",  # 言 speech
            "\u8C9D",  # 貝 shell
        ],
        "numerals": [
            "\u96F6",  # 零 0
            "\u4E8C",  # 二 2
            "\u4E09",  # 三 3
            "\u56DB",  # 四 4
            "\u4E94",  # 五 5
            "\u516D",  # 六 6
            "\u4E03",  # 七 7
            "\u516B",  # 八 8
            "\u4E5D",  # 九 9
            "\u5341",  # 十 10
            "\u767E",  # 百 100
            "\u5343",  # 千 1000
            "\u4E07",  # 万 10000
        ],
    },

    "Katakana": {
        "base": [
            "\u30A2",  # ア a
            "\u30A4",  # イ i
            "\u30A6",  # ウ u
            "\u30A8",  # エ e
            "\u30AA",  # オ o
            "\u30AB",  # カ ka
            "\u30AD",  # キ ki
            "\u30AF",  # ク ku
            "\u30B1",  # ケ ke
            "\u30B3",  # コ ko
            "\u30B5",  # サ sa
            "\u30B7",  # シ shi
            "\u30B9",  # ス su
            "\u30BB",  # セ se
            "\u30BD",  # ソ so
            "\u30BF",  # タ ta
            "\u30C1",  # チ chi
            "\u30C4",  # ツ tsu
            "\u30C6",  # テ te
            "\u30C8",  # ト to
            "\u30CA",  # ナ na
            "\u30CB",  # ニ ni
            "\u30CC",  # ヌ nu
            "\u30CD",  # ネ ne
            "\u30CE",  # ノ no
            "\u30CF",  # ハ ha
            "\u30D2",  # ヒ hi
            "\u30D5",  # フ fu
            "\u30D8",  # ヘ he
            "\u30DB",  # ホ ho
            "\u30DE",  # マ ma
            "\u30DF",  # ミ mi
            "\u30E0",  # ム mu
            "\u30E1",  # メ me
            "\u30E2",  # モ mo
            "\u30E4",  # ヤ ya
            "\u30E6",  # ユ yu
            "\u30E8",  # ヨ yo
            "\u30E9",  # ラ ra
            "\u30EA",  # リ ri
            "\u30EB",  # ル ru
            "\u30EC",  # レ re
            "\u30ED",  # ロ ro
            "\u30EF",  # ワ wa
            "\u30F2",  # ヲ wo
            "\u30F3",  # ン n
        ],
        "dakuten": [
            "\u30AC",  # ガ ga
            "\u30AE",  # ギ gi
            "\u30B0",  # グ gu
            "\u30B2",  # ゲ ge
            "\u30B4",  # ゴ go
            "\u30B6",  # ザ za
            "\u30B8",  # ジ ji
            "\u30BA",  # ズ zu
            "\u30BC",  # ゼ ze
            "\u30BE",  # ゾ zo
            "\u30C0",  # ダ da
            "\u30C2",  # ヂ di
            "\u30C5",  # ヅ du
            "\u30C7",  # デ de
            "\u30C9",  # ド do
            "\u30D0",  # バ ba
            "\u30D3",  # ビ bi
            "\u30D6",  # ブ bu
            "\u30D9",  # ベ be
            "\u30DC",  # ボ bo
        ],
        "handakuten": [
            "\u30D1",  # パ pa
            "\u30D4",  # ピ pi
            "\u30D7",  # プ pu
            "\u30DA",  # ペ pe
            "\u30DD",  # ポ po
        ],
        "small_extended": [
            "\u30A1",  # ァ small a
            "\u30A3",  # ィ small i
            "\u30A5",  # ゥ small u
            "\u30A7",  # ェ small e
            "\u30A9",  # ォ small o
            "\u30C3",  # ッ small tsu
            "\u30E3",  # ャ small ya
            "\u30E5",  # ュ small yu
            "\u30E7",  # ョ small yo
            "\u30EE",  # ヮ small wa
            "\u30F4",  # ヴ vu
            "\u30F5",  # ヵ small ka
            "\u30F6",  # ヶ small ke
        ],
        "signs": [
            "\u30FB",  # ・ middle dot
            "\u30FC",  # ー prolonged sound
            "\u30FD",  # ヽ iteration
            "\u30FE",  # ヾ voiced iteration
            "\u309B",  # ゛ dakuten
            "\u309C",  # ゜ handakuten
        ],
    },
}

# Unicode ranges for script detection
TARGET_SCRIPTS = {
    "Cyrillic":    [(0x0400, 0x04FF), (0x0500, 0x052F)],
    "Katakana":    [(0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    "Devanagari":  [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Arabic":      [(0x0600, 0x06FF), (0x0750, 0x077F),
                    (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "Han":         [(0x4E00, 0x9FFF), (0x3400, 0x4DBF),
                    (0x20000, 0x2A6DF), (0xF900, 0xFAFF)],
    "Bengali":     [(0x0980, 0x09FF)],
    "Tamil":       [(0x0B80, 0x0BFF)],
    "Telugu":      [(0x0C00, 0x0C7F)],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def save_pairwise_similarity(script_name: str, font_names: List[str], embeddings: np.ndarray, out_dir: Path):
    rows = []
    sim = embeddings @ embeddings.T
    for i in range(len(font_names)):
        for j in range(i + 1, len(font_names)):
            rows.append({
                "font_name1": font_names[i],
                "font_name2": font_names[j],
                "similarity": float(sim[i, j]),
            })
    pd.DataFrame(rows).to_csv(out_dir / f"font_similarity_pairs_{script_name}.csv", index=False)


# ===========================================================================
# Flatten reference chars (merge all tiers into one list per script)
# ===========================================================================

def get_flat_reference_chars(script_name: str) -> List[str]:
    """Flatten the tiered reference chars into a single list."""
    tiers = REFERENCE_CHARS.get(script_name, {})
    flat = []
    for tier_name, chars in tiers.items():
        flat.extend(chars)
    return flat


# ===========================================================================
# ViT Feature Extractor
# ===========================================================================

class ViTFeatureExtractor:
    """
    Pretrained ViT-B/16 as a frozen feature extractor.
    Input: 224×224 grayscale glyph image → Output: 768-dim L2-normalized embedding.
    See 10-glyph version for detailed architecture documentation.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        logger.info("Loading pretrained ViT-B/16...")
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.heads.head = nn.Identity()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        logger.info("ViT-B/16 ready (768-dim embeddings)")

    @torch.no_grad()
    def extract_batch(self, images: List[Image.Image], batch_size: int = 64) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = torch.stack([self.transform(img) for img in batch]).to(self.device)
            emb = self.model(tensors).cpu().numpy()
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            emb = emb / norms
            all_embeddings.append(emb)
        return np.vstack(all_embeddings)


# ===========================================================================
# Glyph Rendering
# ===========================================================================

def render_glyph(font_path: str, character: str,
                 canvas_size: int = CANVAS_SIZE,
                 font_size: int = FONT_RENDER_SIZE) -> Optional[Image.Image]:
    """Render a character centered on a grayscale canvas."""
    try:
        pil_font = ImageFont.truetype(font_path, size=font_size)
        img = Image.new("L", (canvas_size, canvas_size), color=255)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), character, font=pil_font)
        if bbox is None or (bbox[2] - bbox[0]) < 3 or (bbox[3] - bbox[1]) < 3:
            return None

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Scale down if glyph is too big for canvas
        if text_w > canvas_size - 10 or text_h > canvas_size - 10:
            scale = min((canvas_size - 10) / text_w, (canvas_size - 10) / text_h)
            new_size = max(10, int(font_size * scale))
            pil_font = ImageFont.truetype(font_path, size=new_size)
            img = Image.new("L", (canvas_size, canvas_size), color=255)
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), character, font=pil_font)
            if bbox is None:
                return None
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

        x = (canvas_size - text_w) // 2 - bbox[0]
        y = (canvas_size - text_h) // 2 - bbox[1]
        draw.text((x, y), character, font=pil_font, fill=0)
        return img
    except Exception:
        return None


def check_font_has_chars(font_path: str, characters: List[str]) -> List[str]:
    """Return which characters the font supports."""
    try:
        font = TTFont(font_path)
        cmap = font.getBestCmap()
        font.close()
        if not cmap:
            return []
        supported = []
        for char in characters:
            # For multi-codepoint sequences, check first codepoint
            cp = ord(char[0])
            if cp in cmap:
                supported.append(char)
        return supported
    except Exception:
        return []


# ===========================================================================
# Font Discovery
# ===========================================================================

def discover_fonts_per_script(fonts_dir: Path) -> Dict[str, List[str]]:
    """Walk Google Fonts and group font files by script."""
    font_extensions = {".ttf", ".otf"}
    script_fonts = defaultdict(list)
    font_files = [f for f in fonts_dir.rglob("*") if f.suffix.lower() in font_extensions]

    logger.info(f"Scanning {len(font_files)} font files...")

    for filepath in font_files:
        try:
            font = TTFont(filepath, fontNumber=0)
            cmap = font.getBestCmap()
            font.close()
            if not cmap:
                continue
            codepoints = set(cmap.keys())
            for script_name, ranges in TARGET_SCRIPTS.items():
                count = sum(1 for cp in codepoints for s, e in ranges if s <= cp <= e)
                if count >= MIN_CODEPOINT_COVERAGE:
                    script_fonts[script_name].append(str(filepath))
        except Exception:
            continue

    for script, fonts in script_fonts.items():
        logger.info(f"  {script}: {len(fonts)} fonts")
    return dict(script_fonts)


# ===========================================================================
# Diversity Metrics
# ===========================================================================

def compute_diversity_metrics(embeddings: np.ndarray) -> Dict:
    """Compute pairwise cosine diversity and PCA dimensionality."""
    n = embeddings.shape[0]
    if n < 2:
        return {"mean_cosine_distance": 0.0, "std_cosine_distance": 0.0,
                "embedding_spread": 0.0, "effective_dimensions": 0, "n_embeddings": n}

    sim = embeddings @ embeddings.T
    idx = np.triu_indices(n, k=1)
    dists = 1.0 - sim[idx]

    try:
        centered = embeddings - embeddings.mean(axis=0)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        explained = S ** 2 / np.sum(S ** 2)
        eff_dims = int(np.searchsorted(np.cumsum(explained), 0.90) + 1)
    except:
        eff_dims = 0

    return {
        "mean_cosine_distance": float(np.mean(dists)),
        "std_cosine_distance": float(np.std(dists)),
        "embedding_spread": float(np.mean(np.std(embeddings, axis=0))),
        "effective_dimensions": eff_dims,
        "n_embeddings": n,
    }


# ===========================================================================
# Main Pipeline
# ===========================================================================

def run_pipeline(fonts_dir: Path) -> pd.DataFrame:
    script_fonts = discover_fonts_per_script(fonts_dir)
    extractor = ViTFeatureExtractor(device="cuda" if torch.cuda.is_available() else "cpu")

    results = []

    for script_name, font_paths in script_fonts.items():
        ref_chars = get_flat_reference_chars(script_name)
        if not ref_chars:
            continue

        logger.info(f"\nProcessing {script_name} ({len(font_paths)} fonts, "
                     f"{len(ref_chars)} reference chars)...")

        # Deterministic sampling
        font_paths = sorted(font_paths)
        np.random.seed(42)
        if len(font_paths) > MAX_FONTS_PER_SCRIPT:
            sampled = list(np.random.choice(font_paths, MAX_FONTS_PER_SCRIPT, replace=False))
            logger.info(f"  Sampled {MAX_FONTS_PER_SCRIPT}/{len(font_paths)} fonts")
        else:
            sampled = font_paths

        font_avg_embeddings = []
        font_names = []
        total_glyphs = 0
        tier_stats = defaultdict(int)  # track which tiers rendered

        for font_path in sampled:
            supported = check_font_has_chars(font_path, ref_chars)
            if len(supported) < 5:
                continue

            images = []
            for char in supported:
                img = render_glyph(font_path, char)
                if img is not None:
                    images.append(img)

            if len(images) < 5:
                continue

            embs = extractor.extract_batch(images)
            total_glyphs += len(images)

            font_avg = np.mean(embs, axis=0)
            font_avg = font_avg / np.linalg.norm(font_avg)
            font_avg_embeddings.append(font_avg)
            font_names.append(Path(font_path).stem)

        if len(font_avg_embeddings) < 2:
            logger.warning(f"  Insufficient fonts, skipping")
            continue

        matrix = np.vstack(font_avg_embeddings)
        save_pairwise_similarity(script_name, font_names, matrix, FONT_SIMILARITY_DIR)
        logger.info(f"  {len(font_avg_embeddings)} fonts, {total_glyphs} glyphs rendered")

        metrics = compute_diversity_metrics(matrix)

        # Save embeddings
        np.save(EMBEDDINGS_DIR / f"{script_name}_embeddings.npy", matrix)

        results.append({
            "script": script_name,
            "total_fonts": len(font_paths),
            "fonts_analyzed": len(font_avg_embeddings),
            "reference_chars": len(ref_chars),
            "glyphs_rendered": total_glyphs,
            **metrics,
        })

        logger.info(f"  Diversity: cosine_dist={metrics['mean_cosine_distance']:.4f}, "
                     f"eff_dims={metrics['effective_dimensions']}")

    return pd.DataFrame(results)


def compute_diversity_index(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    col = "mean_cosine_distance"
    cmin, cmax = result[col].min(), result[col].max()
    if cmax - cmin == 0:
        result["diversity_index"] = 0.5
    else:
        result["diversity_index"] = (result[col] - cmin) / (cmax - cmin)
    return result.sort_values("diversity_index")


def print_report(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("SCRIPT DIVERSITY INDEX — ViT-B/16 (100-glyph edition)")
    print("=" * 80)

    print("\n📊 Embedding analysis:\n")
    cols = ["script", "fonts_analyzed", "reference_chars", "glyphs_rendered",
            "mean_cosine_distance", "effective_dimensions"]
    print(df[cols].to_string(index=False, float_format="%.4f"))

    print("\n\n📐 Diversity Index:\n")
    dcols = ["script", "diversity_index", "mean_cosine_distance", "embedding_spread"]
    print(df[dcols].to_string(index=False, float_format="%.4f"))

    print("\n\n🔍 Interpretation:")
    print("  D → 1.0 = High diversity (fonts look very different)")
    print("  D → 0.0 = Low diversity (fonts all look similar)")
    top = df.loc[df["diversity_index"].idxmax()]
    bot = df.loc[df["diversity_index"].idxmin()]
    print(f"\n  Most diverse:  {top['script']} (D = {top['diversity_index']:.3f})")
    print(f"  Least diverse: {bot['script']} (D = {bot['diversity_index']:.3f})")
    print("\n" + "=" * 80)


def main():
    if not GOOGLE_FONTS_DIR.exists():
        logger.error(f"Google Fonts not found at {GOOGLE_FONTS_DIR}")
        logger.error("Clone it: git clone --depth 1 https://github.com/google/fonts")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    logger.info("Starting ViT Diversity Pipeline (100-glyph edition)...")
    raw = run_pipeline(GOOGLE_FONTS_DIR)

    if raw.empty:
        logger.error("No results!")
        sys.exit(1)

    result = compute_diversity_index(raw)
    print_report(result)

    result.to_csv(OUTPUT_DIR / "diversity_index_results.csv", index=False)
    logger.info(f"Saved → {OUTPUT_DIR / 'diversity_index_results.csv'}")

    summary = result[["script", "diversity_index", "mean_cosine_distance",
                       "fonts_analyzed", "reference_chars", "glyphs_rendered"]].copy()
    summary.to_csv(OUTPUT_DIR / "diversity_index_summary.csv", index=False)
    logger.info(f"Saved → {OUTPUT_DIR / 'diversity_index_summary.csv'}")


if __name__ == "__main__":
    main()
