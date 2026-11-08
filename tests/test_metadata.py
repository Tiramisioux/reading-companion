import unittest

from reading_companion.metadata import (PageMeta, PrintRow, chapter_number_token, consolidate, folio_token,
                                        pretty_title, read_page)


def row(where, x0, y0, x1, y1, h, words):
    r = PrintRow(where, x0, y0, x1, y1, h)
    r.words = [(w, 0.9, 0, 0, 10, 10) for w in words]
    r.img_w = 100
    return r


class TokenTest(unittest.TestCase):
    def test_folio(self):
        self.assertEqual(folio_token("224"), ("224", 224))
        self.assertEqual(folio_token("2O3"), ("203", 203))
        self.assertEqual(folio_token("xii"), ("xii", 12))
        self.assertIsNone(folio_token("I"))
        self.assertIsNone(folio_token("AM"))

    def test_chapter_number(self):
        self.assertEqual(chapter_number_token("IO"), "10")  # old-style figures read as letters
        self.assertEqual(chapter_number_token("Chapter 10"), "10")
        self.assertEqual(chapter_number_token("CHAPTER TEN"), "10")
        self.assertEqual(chapter_number_token("Kapitel 3"), "3")
        self.assertEqual(chapter_number_token("XII"), "XII")
        self.assertIsNone(chapter_number_token("Am I Here?"))

    def test_pretty_title(self):
        self.assertEqual(pretty_title("AM I HERE?"), "Am I Here?")
        self.assertEqual(pretty_title("W A Y F I N D I N G"), "Wayfinding")
        self.assertEqual(pretty_title("FROM A TO B AND BACK AGAIN"), "From A to B and Back Again")


class ReadPageTest(unittest.TestCase):
    def test_running_head_with_folio_left(self):
        rows = [row("top", 300, 40, 340, 60, 18, ["224"]), row("top", 700, 40, 960, 60, 18, ["WAYFINDING"])]
        m = read_page(rows, 58.0, 320, 1350, 118, 1900, letter_h=17)
        self.assertEqual((m.page_number, m.side, m.running_head), ("224", "left", "Wayfinding"))

    def test_running_head_with_folio_right_in_one_group(self):
        rows = [row("top", 500, 370, 1100, 400, 19, ["EPILOGUE", "221"])]
        m = read_page(rows, 47.0, 250, 1110, 446, 1950, letter_h=17)
        self.assertEqual((m.page_number, m.side, m.running_head), ("221", "right", "Epilogue"))

    def test_chapter_opening(self):
        rows = [row("top", 770, 320, 820, 355, 30, ["IO"]), row("top", 650, 430, 930, 470, 29, ["Am", "I", "Here?"])]
        m = read_page(rows, 55.0, 290, 1270, 712, 1840, letter_h=17)
        self.assertTrue(m.chapter_opening)
        self.assertEqual((m.chapter_number, m.chapter_title), ("10", "Am I Here?"))
        self.assertIsNone(m.page_number)

    def test_footer_folio(self):
        rows = [row("bottom", 700, 1950, 740, 1975, 18, ["57"])]
        m = read_page(rows, 55.0, 290, 1270, 200, 1900, letter_h=17)
        self.assertEqual((m.page_number, m.page_number_source), ("57", "footer"))


class FolderTest(unittest.TestCase):
    def sample(self):
        return [PageMeta(), PageMeta("224", 224, "header", "left", "Wayfinding"),
                PageMeta("221", 221, "header", "right", "Epilogue"), PageMeta("220", 220, "header", "left", "Wayfinding"),
                PageMeta("203", 203, "header", "right", "Am I Here?"), PageMeta("224", 224, "header", "left", None),
                PageMeta("202", 202, "header", "left", "Wayfind1ng"),  # an OCR slip
                PageMeta(chapter_opening=True, chapter_number="10", chapter_title="Am I Here?", chapter_source="chapter_opening")]

    def test_book_title_and_chapters(self):
        pages = self.sample()
        book = consolidate(pages, folder_name="scans")
        self.assertEqual(book["title"], "Wayfinding")
        self.assertEqual((pages[2].chapter_title, pages[2].chapter_source), ("Epilogue", "running_head"))
        self.assertEqual((pages[3].chapter_title, pages[3].chapter_source), ("Epilogue", "adjacent_page"))
        self.assertEqual((pages[4].chapter_number, pages[4].chapter_title), ("10", "Am I Here?"))
        self.assertEqual((pages[6].chapter_number, pages[6].chapter_source), ("10", "adjacent_page"))
        self.assertIsNone(pages[1].chapter_source)        # 224: nothing in the folder says
        self.assertIsNone(pages[0].chapter_source)        # no running head, no page number
        self.assertEqual(pages[5].running_head, "Wayfinding")  # same page photographed twice

    def test_folder_name_names_the_book(self):
        pages = [PageMeta("10", 10, "header", "left", "Wayfinding")]
        self.assertEqual(consolidate(pages, folder_name="Wayfinding (2019)")["title_source"], "folder_name")

    def test_no_guessing_without_evidence(self):
        pages = [PageMeta("33", 33, "header", "right", "Some Chapter")]
        book = consolidate(pages, folder_name="scans")
        self.assertIsNone(book["title"])
        self.assertIsNone(pages[0].chapter_source)  # could be the book title: not claimed as a chapter


if __name__ == "__main__":
    unittest.main()
