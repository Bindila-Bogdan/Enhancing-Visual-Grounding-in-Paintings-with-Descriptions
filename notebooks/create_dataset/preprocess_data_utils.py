import re
import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords

STOP_WORDS = stopwords.words("english")
MIN_DESCRIPTION_WORD_COUNT = 20
MIN_YEAR = 0
MAX_YEAR = 2026


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
