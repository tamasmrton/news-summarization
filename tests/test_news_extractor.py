from datetime import datetime
import pytest
from unittest.mock import Mock, patch
from bs4 import BeautifulSoup
from news_summarization.news_extractor import NewsExtractor

MOCK_SITEMAP_CONTENT = """
<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url>
            <loc>https://example.com/2023/08/13/article1</loc>
            <lastmod>2023-08-13T12:00:00+00:00</lastmod>
        </url>
        <url>
            <loc>https://example.com/2023/08/12/article2</loc>
            <lastmod>2023-08-12T10:00:00+00:00</lastmod>
        </url>
        <url>
            <loc>https://example.com/2023/08/13/article3</loc>
        </url>
        <url>
            <loc>https://example.com/2023/08/12/article4</loc>
        </url>
    </urlset>
"""

@pytest.fixture
def news_extractor():
    return NewsExtractor(base_url='https://example.com', date='2023-08-13', date_format='%Y-%m-%d')

def test_init(news_extractor):
    assert news_extractor.base_url == 'https://example.com'
    assert news_extractor.date == '2023-08-13'
    assert news_extractor.date_format == '%Y-%m-%d'
    assert news_extractor.date_formatted == datetime.strptime('2023-08-13', '%Y-%m-%d')

def test_get_sitemaps(news_extractor, mocker):
    # Mock the requests.get method and its response
    response_mock = Mock()
    response_mock.text = "sitemap: https://example.com/sitemap.xml"
    mocker.patch('requests.get', return_value=response_mock)

    sitemaps = news_extractor.get_sitemaps()

    assert len(sitemaps) == 1
    assert sitemaps[0] == 'https://example.com/sitemap.xml'

def test_parse_sitemap_with_lastmod(news_extractor):
    soup = BeautifulSoup(MOCK_SITEMAP_CONTENT, 'xml')
    links = news_extractor._parse_sitemap_with_lastmod(soup)

    assert len(links) == 1

def test_parse_sitemap_without_lastmod(news_extractor):
    soup = BeautifulSoup(MOCK_SITEMAP_CONTENT, 'xml')
    links = news_extractor._parse_sitemap_without_lastmod(soup)

    assert len(links) == 2

if __name__ == '__main__':
    pytest.main()
