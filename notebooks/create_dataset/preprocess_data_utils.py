import re
import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords

# constants for data retrieval filtering
STOP_WORDS = stopwords.words("english")
MIN_DESCRIPTION_WORD_COUNT = 20
MIN_YEAR = 0
MAX_YEAR = 2026
MIN_YEAR_FILTERING = 1300

MET_KEPT_OBJECT_TYPES = ["Painting", "Drawing", "Painting, drawing", "Painting, sketch", "Print"]

WIKIART_KEPT_TYPES = [
    "bijinga",
    "capriccio",
    "cityscape",
    "cloudscape",
    "design",
    "icon",
    "interior",
    "landscape",
    "marina",
    "miniature",
    "painting",
    "pastorale",
    "portrait",
    "still life",
    "veduta",
    "yakusha-e",
]

WIKIART_LEFT_OUT_TYPES = [
    "abstract",
    "advertisement",
    "architecture",
    "calligraphy",
    "caricature",
    "digital",
    "graffiti",
    "furniture",
    "jewelry",
    "illustration",
    "installation",
    "mosaic",
    "mural",
    "ornament",
    "performance",
    "photo",
    "poster",
    "sculpture",
    "sketch and study",
    "tapestry",
    "utensil",
    "vanitas",
    "video",
]

WIKIART_LEFT_OUT_STYLES = ["abstract", "cubism", "cubo-futurism", "dada", "futurism", "orphism"]

WIKIART_LEFT_OUT_MEDIA = [
    "engraving",
    "photography",
    "crayon",
    "lithography",
    "collage",
    "etching",
    "japanese paper",
]

WGA_KEPT_TECHNIQUES = [
    "oil panel",
    "oil wood",
    "oil oak panel",
    "oil copper",
    "wood",
    "oil oak",
    "oil tempera limewood",
    "oil tempera wood",
    "oil tempera red beechwood",
    "oil poplar panel",
    "oil tempera panel",
    "oil tempera panel",
    "poplar panel",
]

FILTERED_OUT_TYPES = ["trompe-l'œil", "sculpture", "design", "quadratura", "poster"]

FILTERED_OUT_STYLES = [
    "mosan art",
    "muralism",
    "conceptual art, pop art",
    "early christian",
    "new kingdom",
    "3rd intermediate period",
    "photorealism",
    "cubo-expressionism",
]

FILTERED_OUT_COARSE_TYPES = ["sketch and study", "interior"]

# constants for normalizing the styles and types of paintings
REPLACEMENTS = {
    "middle byzantine (c. 850–1204)": "byzantine",
    "novgorod school of icon painting": "",
    "late byzantine/palaeologan renaissance (c. 1261–1453)": "byzantine",
    "macedonian renaissance (867–1056)": "",
    "ink and wash painting": "",
    "\xa0": " ",
    " (modern)": "",
    " painting": "",
    " (nu)": "",
}

FINE_GRAINED_TYPES_MAPPING = {
    "icon": "religious",
    "architecture": "veduta",
    "vanitas": "portrait",
    "wildlife": "animal",
    "illustration": None,
    "yakusha-e": None,
    "panorama": None,
    "caricature": None,
}

STYLES_MAPPING = {
    "social realism": "realism",
    "new realism": "realism",
    "contemporary realism": "realism",
    "american realism": "realism",
    "socialist realism": "realism",
    "hyper-realism": "realism",
    "neo-rococo": "rococo",
    "new european painting": None,
    "art brut": None,
    "new european": None,
    "op art": "pop art",
    "analytical realism": None,
    "kanō school style": None,
    "lowbrow art": None,
    "mughal": None,
    "purism": None,
    "neo-suprematism": None,
    "modernism": None,
    "intimism": None,
    "naturalism": None,
    "neo-expressionism": None,
    "metaphysical art": None,
    "art deco": None,
    "kitsch": None,
    "neo-baroque": None,
    "precisionism": None,
    "tonalism": None,
    "postcolonial art": None,
    "ukiyo-e": None,
    "gothic": None,
    "biedermeier": None,
    "synthetism": None,
    "luminism": None,
    "native art": None,
    "verism": None,
    "feminist art": None,
    "safavid period": None,
    "japonism": None,
    "proto renaissance": None,
    "international gothic": None,
    "byzantine": None,
    "": None,
}


def clean_artist_name(artist):
    first_artist_name = re.sub(r"\([^)]*\)", "", artist.lower()).split("|")[0]
    artist_wo_punctuation = re.sub(r"[.,\-!?;:()\[\]{}]", " ", first_artist_name).strip()
    artist_wo_multiple_spaces = re.sub(r"\s+", " ", artist_wo_punctuation).strip()

    return artist_wo_multiple_spaces


def clean_title_name(title):
    title_wo_punctuation = re.sub(
        r"[.,\-!?;:()\[\]{}]", " ", title.lower().replace("\xa0", " ")
    ).strip()
    title_wo_multiple_spaces = re.sub(r"\s+", " ", title_wo_punctuation).strip().split(" ")
    title_wo_stop_words = " ".join(
        [word for word in title_wo_multiple_spaces if word not in STOP_WORDS]
    )

    return title_wo_stop_words


def clean_description(description):
    description_wo_links = re.compile(r"https?://\S+|www\.\S+").sub("", description)

    description_wo_punctuation = re.sub(
        r"[.,\–\-!?;:()\[\]{}\'\"/]", " ", description_wo_links.lower()
    )

    description_wo_numbers = re.sub(r"\d+", "", description_wo_punctuation)

    description_wo_multiple_spaces = re.sub(r"\s+", " ", description_wo_numbers).strip().split(" ")

    descriptions_wo_stop_words = " ".join(
        [
            word
            for word in description_wo_multiple_spaces
            if word not in STOP_WORDS and len(word) > 1
        ]
    )

    return descriptions_wo_stop_words


def clean_genre(genre):
    cleaned_genre = genre.replace(" painting", "").replace(" (nu)", "").strip().lower()

    if cleaned_genre == "none":
        return None
    else:
        return cleaned_genre


def clean_style(style):
    cleaned_style = style.replace(" painting", "").replace("\xa0", " ").strip().lower()

    if clean_style == "none":
        return None
    else:
        return cleaned_style


def clean_date(year, index=0):
    cleaned_year = re.findall(r"\b\d{4}\b", year.lower())

    if len(cleaned_year) == 0:
        return None
    else:
        return int(cleaned_year[index])


def rearrange_artist_name(name):
    divided_name = name.split(", ")

    if len(divided_name) == 1:
        return divided_name[0]

    elif len(divided_name) == 2:
        return divided_name[1] + " " + divided_name[0]

    else:
        return divided_name[1] + " " + divided_name[0] + " " + " ".join(divided_name[2:])


def is_same_artist(searched_artist, current_artist):
    search_words = searched_artist.split(" ")
    target_words = current_artist.split(" ")

    return all(word in target_words for word in search_words) or all(
        word in search_words for word in target_words
    )


def is_same_painting(searched_title, searched_artist, title, artist):
    found_artist = is_same_artist(searched_artist, artist)
    found_title = searched_title in title or title in searched_title

    return found_artist and found_title


def replace_text(text):
    if text is None:
        return None

    for to_replace, replacement in REPLACEMENTS.items():
        text = text.replace(to_replace, replacement)

    return text


def sort_elements(properties, first_name, second_name):
    updated_properties = [
        replace_text(properties[first_name]),
        replace_text(properties[second_name]),
    ]

    if updated_properties[0] is None:
        return [updated_properties[1], None]

    elif updated_properties[1] is None:
        return [updated_properties[0], None]

    else:
        return sorted([updated_properties[0], updated_properties[1]])
