"""Text for synthetic samples.

Modes (each fights a specific failure we measured on LightOnOCR):
  natural     real Kazakh sentences (Wikipedia)                       -> normal reading
  misspelled  Kazakh letters swapped for Russian lookalikes (Қ->К ...) -> copy what is printed, don't "fix" it
  pseudo      made-up words + real words with one confusable letter   -> no word to guess: must read ұ/ү, ң/н, ғ/г
  entities    names, patronymics, addresses, dates, IDs, amounts       -> rare words, where guessing fails most
"""

from __future__ import annotations

import random
from pathlib import Path

from kzocr.common import DATA_DIR, KZ_TO_RU_LOOKALIKE

TEXT_DIR = DATA_DIR / "text"
LOWER = "аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя"
KZ_LOWER = "әғқңөұүһі"
VOWELS = set("аәеёиоөуұүыіэюя")
# Visually confusable pairs used to "mutate" real words into near-words.
CONFUSABLE = {"у": "ұү", "ұ": "үу", "ү": "ұу", "н": "ң", "ң": "н", "г": "ғ", "ғ": "г", "к": "қ", "қ": "к",
              "о": "ө", "ө": "о", "а": "ә", "ә": "а", "и": "і", "і": "иы", "х": "һ", "һ": "х", "ы": "і"}
CITIES = ["Алматы", "Астана", "Шымкент", "Қарағанды", "Ақтөбе", "Тараз", "Павлодар", "Өскемен", "Семей",
          "Атырау", "Қостанай", "Қызылорда", "Орал", "Петропавл", "Ақтау", "Түркістан", "Көкшетау",
          "Талдықорған", "Екібастұз", "Жезқазған", "Теміртау", "Қонаев", "Балқаш", "Сәтбаев", "Жаңаөзен"]
STREETS = ["Абай", "Төле би", "Бөгенбай батыр", "Қабанбай батыр", "Жамбыл", "Әйтеке би", "Қазыбек би",
           "Сәтбаев", "Бұқар жырау", "Әл-Фараби", "Құрманғазы", "Желтоқсан", "Тәуелсіздік", "Мәңгілік Ел",
           "Сарыарқа", "Қонаев", "Ғабит Мүсірепов", "Шоқан Уәлиханов", "Мұхтар Әуезов", "Ыбырай Алтынсарин"]
MONTHS = ["қаңтар", "ақпан", "наурыз", "сәуір", "мамыр", "маусым", "шілде", "тамыз", "қыркүйек", "қазан",
          "қараша", "желтоқсан"]
FIRST = ["Айгерім", "Нұрсұлтан", "Ғалия", "Ерғали", "Жәнібек", "Сәуле", "Құралай", "Мөлдір", "Әсел", "Дәурен",
         "Бауыржан", "Гүлнұр", "Ұлжан", "Тоғжан", "Арман", "Қуаныш", "Нұрлан", "Шолпан", "Айбек", "Ләззат",
         "Ғабит", "Әлихан", "Мұхтар", "Жұлдыз", "Ақбота", "Өмірзақ", "Серік", "Бекзат", "Дана", "Әйгерім"]
SURNAMES = ["Нұрәлиев", "Әбдіғалиев", "Сұлтанбеков", "Қасымов", "Жұмабаев", "Төлегенов", "Мұқанов",
            "Сәрсенбаев", "Құдайбергенов", "Ғаббасов", "Өтебаев", "Бекмұхамедов", "Ыбыраев", "Әуезов",
            "Есенғазиев", "Қожахметов", "Тұрсынбаев", "Шәкенов", "Үсенов", "Ахметов"]
PHRASES = ["АНЫҚТАМА", "ХАБАРЛАМА", "БҰЙРЫҚ", "ӨТІНІШ", "ШЕШІМ", "ХАТТАМА", "ҚҰЖАТТАР ТІЗІЛІМІ",
           "Берілген күні", "Жауапты маман", "Мөр орны", "Басшы", "Орынбасары", "Мекенжайы", "Туған күні",
           "Тегі, аты, әкесінің аты", "Құжат нөмірі", "Анықтама берілген жері", "Қолы", "Келісілді",
           "Бекітемін", "Негізгі мәліметтер", "Ескерту", "Қосымша", "Тіркеу нөмірі"]
FIELDS = ["Тегі", "Аты", "Әкесінің аты", "Туған күні", "Туған жері", "Мекенжайы", "ЖСН", "Телефон",
          "Құжат нөмірі", "Берілген күні", "Лауазымы", "Ұйым", "Сомасы", "Мерзімі", "Мәртебесі"]


class TextSource:
    def __init__(self, text_dir: Path = TEXT_DIR):
        sent_file = text_dir / "kk_sentences.txt"
        if not sent_file.exists():
            raise FileNotFoundError(f"{sent_file} missing - run: python data_scripts/download_wikipedia_kk.py")
        self.sentences = [s for s in sent_file.read_text(encoding="utf-8").splitlines() if s]
        self.words, self.word_weights = [], []
        vocab = text_dir / "kk_vocab.tsv"
        for line in vocab.read_text(encoding="utf-8").splitlines():
            w, c = line.split("\t")
            self.words.append(w)
            self.word_weights.append(int(c) ** 0.5)  # flatten the frequency curve -> more rare words
        names = text_dir / "kk_names.txt"
        self.names = [n for n in names.read_text(encoding="utf-8").splitlines() if n] if names.exists() else []

    # ---------------------------------------------------------------- building blocks
    def _natural(self, rng: random.Random, max_chars: int) -> str:
        out = ""
        for _ in range(8):
            s = rng.choice(self.sentences)
            cand = (out + " " + s).strip()
            if len(cand) > max_chars:
                if not out:  # single long sentence: cut at a word boundary
                    out = s[:max_chars].rsplit(" ", 1)[0] if " " in s[:max_chars] else s[:max_chars]
                break
            out = cand
            if len(out) > max_chars * 0.6 and rng.random() < 0.5:
                break
        return out

    def _misspell(self, text: str, rng: random.Random) -> str:
        p = rng.uniform(0.3, 1.0)
        out = "".join(KZ_TO_RU_LOOKALIKE[c] if c in KZ_TO_RU_LOOKALIKE and rng.random() < p else c for c in text)
        if out == text:  # make sure at least one letter changed
            idx = [i for i, c in enumerate(text) if c in KZ_TO_RU_LOOKALIKE]
            if idx:
                i = rng.choice(idx)
                out = text[:i] + KZ_TO_RU_LOOKALIKE[text[i]] + text[i + 1:]
        return out

    def made_up_word(self, rng: random.Random) -> str:
        n = rng.randint(3, 9)
        w = []
        for i in range(n):
            pool = KZ_LOWER if rng.random() < 0.45 else LOWER
            c = rng.choice(pool)
            if i and (c in VOWELS) == (w[-1] in VOWELS) and rng.random() < 0.5:  # roughly pronounceable
                c = rng.choice([x for x in LOWER if (x in VOWELS) != (w[-1] in VOWELS)])
            w.append(c)
        w = "".join(w)
        r = rng.random()
        return w.upper() if r < 0.1 else w.capitalize() if r < 0.3 else w

    def mutated_word(self, rng: random.Random) -> str:
        w = rng.choices(self.words, weights=self.word_weights)[0]
        idx = [i for i, c in enumerate(w.lower()) if c in CONFUSABLE]
        if not idx:
            return w
        i = rng.choice(idx)
        c = w[i]
        rep = rng.choice(CONFUSABLE[c.lower()])
        return w[:i] + (rep.upper() if c.isupper() else rep) + w[i + 1:]

    def _pseudo(self, rng: random.Random, max_chars: int) -> str:
        words, length = [], 0
        while True:
            r = rng.random()
            w = (self.made_up_word(rng) if r < 0.45 else self.mutated_word(rng) if r < 0.75
                 else rng.choices(self.words, weights=self.word_weights)[0])
            if length + len(w) + 1 > max_chars and words:
                break
            words.append(w)
            length += len(w) + 1
            if length > max_chars * 0.5 and rng.random() < 0.25:
                break
        if rng.random() < 0.3:
            words[-1] += rng.choice(".,;:")
        return " ".join(words)

    def person(self, rng: random.Random) -> str:
        first = rng.choice(FIRST + self.names[:2000]) if self.names else rng.choice(FIRST)
        sur = rng.choice(SURNAMES)
        female = rng.random() < 0.5
        if female and sur.endswith(("ов", "ев", "ин")):
            sur += "а"
        father = rng.choice(FIRST)
        if rng.random() < 0.6:
            pat = father + ("қызы" if female else "ұлы")
        else:
            base = father.rstrip("аы")
            pat = base + ("овна" if female else "ович")
        r = rng.random()
        if r < 0.15:
            return f"{sur.upper()} {first.upper()} {pat.upper()}"
        if r < 0.3:
            return f"{first[0]}. {sur}"
        return f"{sur} {first} {pat}"

    def entity(self, rng: random.Random) -> str:
        r = rng.random()
        d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1950, 2026)
        if r < 0.22:
            return self.person(rng)
        if r < 0.40:
            city, street = rng.choice(CITIES), rng.choice(STREETS)
            forms = [f"{rng.randint(10, 99):02d}00{rng.randint(10, 99)}, {city} қаласы, {street} көшесі, {rng.randint(1, 300)}",
                     f"{city} қ., {street} к-сі, {rng.randint(1, 200)}-үй, {rng.randint(1, 150)}-пәтер",
                     f"{city} облысы, {rng.choice(CITIES)} ауданы, {street} даңғылы {rng.randint(1, 99)}"]
            return rng.choice(forms)
        if r < 0.55:
            forms = [f"{d:02d}.{m:02d}.{y}", f"{d:02d}.{m:02d}.{y} ж.", f"{y} жылғы {d} {MONTHS[m - 1]}",
                     f"«{d:02d}» {MONTHS[m - 1]} {y} ж."]
            return rng.choice(forms)
        if r < 0.70:
            forms = [f"№ {rng.randint(10**9, 10**14 - 1)}", f"ЖСН: {rng.randint(10**11, 10**12 - 1)}",
                     f"Тел.: 8({rng.choice(['727', '7172', '7252', '701', '777', '705'])}) {rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(10, 99)}",
                     f"{rng.randint(1, 999)} {rng.randint(0, 999):03d} теңге", f"Тіркеу № {y}-{rng.randint(100, 9999)}"]
            return rng.choice(forms)
        if r < 0.85:
            return f"{rng.choice(FIELDS)}: {self.entity(rng) if rng.random() < 0.7 else self.mutated_word(rng)}"
        return rng.choice(PHRASES) + (f" · {rng.randint(1, 9999):04d}" if rng.random() < 0.3 else "")

    def _entities(self, rng: random.Random, max_chars: int) -> str:
        out = self.entity(rng)
        while rng.random() < 0.35:
            nxt = self.entity(rng)
            if len(out) + len(nxt) + 2 > max_chars:
                break
            out += rng.choice([", ", "; ", " — ", "  "]) + nxt
        return out[:max_chars]

    # ---------------------------------------------------------------- public API
    def text(self, rng: random.Random, mode: str, max_chars: int = 80) -> tuple[str, str | None]:
        """Returns (printed_text, correct_text_if_misspelled_else_None)."""
        if mode == "natural":
            return self._natural(rng, max_chars), None
        if mode == "misspelled":
            orig = self._natural(rng, max_chars)
            return self._misspell(orig, rng), orig
        if mode == "pseudo":
            return self._pseudo(rng, max_chars), None
        if mode == "entities":
            return self._entities(rng, max_chars), None
        raise ValueError(mode)
