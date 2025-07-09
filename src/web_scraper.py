"""
Web scraper module for collecting bioengineering and biology educational content.
"""

import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import os
import logging
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse
import re
from dataclasses import dataclass
from config import scraping_config, data_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScrapedContent:
    """Structure for scraped content"""

    url: str
    title: str
    content: str
    source: str
    difficulty_level: str = "intermedio"  # Default level
    topic: str = ""


class WebScraper:
    """Web scraper for bioengineering and biology content"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(scraping_config.headers)
        self.scraped_data = []

    def scrape_wikipedia_biology(
        self, topics: List[str], language: str = "es"
    ) -> List[ScrapedContent]:
        """
        Scrape Wikipedia articles for biology topics

        Args:
            topics: List of biology topics to scrape
            language: Language for Wikipedia (default: Spanish)

        Returns:
            List of ScrapedContent objects
        """
        base_url = f"https://{language}.wikipedia.org/wiki/"
        scraped_content = []

        for topic in topics:
            try:
                url = base_url + topic.replace(" ", "_")
                response = self.session.get(url, timeout=scraping_config.timeout)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, "html.parser")

                # Extract title
                title_element = soup.find("h1", class_="firstHeading")
                title = title_element.text if title_element else topic

                # Extract main content
                content_div = soup.find("div", class_="mw-parser-output")
                if content_div:
                    # Remove unwanted elements
                    for element in content_div.find_all(
                        ["table", "div", "sup", "style", "script"]
                    ):
                        element.decompose()

                    # Extract text from paragraphs
                    paragraphs = content_div.find_all("p")
                    content = " ".join(
                        [
                            p.get_text().strip()
                            for p in paragraphs
                            if p.get_text().strip()
                        ]
                    )

                    # Clean content
                    content = self._clean_text(content)

                    if len(content) > 200:  # Only keep substantial content
                        scraped_content.append(
                            ScrapedContent(
                                url=url,
                                title=title,
                                content=content,
                                source="wikipedia",
                                difficulty_level=self._estimate_difficulty(content),
                                topic=topic,
                            )
                        )

                        logger.info(f"Scraped: {title} ({len(content)} chars)")

                time.sleep(scraping_config.delay_between_requests)

            except Exception as e:
                logger.error(f"Error scraping {topic}: {str(e)}")
                continue

        return scraped_content

    def scrape_khan_academy_biology(self) -> List[ScrapedContent]:
        """
        Scrape Khan Academy biology content (Spanish)
        Note: This is a simplified version - real implementation would need
        to handle dynamic content and authentication
        """
        # This would need to be implemented with Selenium for dynamic content
        # For now, return empty list
        logger.info("Khan Academy scraping not implemented - requires Selenium")
        return []

    def scrape_educational_sites(self, urls: List[str]) -> List[ScrapedContent]:
        """
        Scrape educational content from provided URLs

        Args:
            urls: List of URLs to scrape

        Returns:
            List of ScrapedContent objects
        """
        scraped_content = []

        for url in urls:
            try:
                response = self.session.get(url, timeout=scraping_config.timeout)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, "html.parser")

                # Extract title
                title_element = soup.find("title") or soup.find("h1")
                title = title_element.get_text().strip() if title_element else "Unknown"

                # Extract content from common content containers
                content_selectors = [
                    "article",
                    "main",
                    ".content",
                    ".post-content",
                    ".entry-content",
                    ".article-content",
                    "p",
                ]

                content = ""
                for selector in content_selectors:
                    elements = soup.select(selector)
                    if elements:
                        content = " ".join(
                            [elem.get_text().strip() for elem in elements]
                        )
                        break

                content = self._clean_text(content)

                if len(content) > 200:
                    scraped_content.append(
                        ScrapedContent(
                            url=url,
                            title=title,
                            content=content,
                            source=urlparse(url).netloc,
                            difficulty_level=self._estimate_difficulty(content),
                        )
                    )

                    logger.info(f"Scraped: {title} from {urlparse(url).netloc}")

                time.sleep(scraping_config.delay_between_requests)

            except Exception as e:
                logger.error(f"Error scraping {url}: {str(e)}")
                continue

        return scraped_content

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep Spanish accents
        text = re.sub(r"[^\w\s\.\,\;\:\!\?\-\(\)áéíóúüñÁÉÍÓÚÜÑ]", "", text)

        # Remove very short sentences
        sentences = text.split(".")
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        return ". ".join(sentences)

    def _estimate_difficulty(self, text: str) -> str:
        """
        Estimate difficulty level based on text characteristics

        Args:
            text: Text to analyze

        Returns:
            Difficulty level (principiante, intermedio, experto)
        """
        # Simple heuristic based on sentence length and technical terms
        sentences = text.split(".")
        avg_sentence_length = (
            sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        )

        # Technical terms that indicate higher difficulty
        technical_terms = [
            "proteína",
            "enzima",
            "mitocondria",
            "nucleótido",
            "genoma",
            "fosforilación",
            "transcripción",
            "traducción",
            "cromosoma",
            "metabolismo",
            "homeostasis",
            "transducción",
            "citoplasma",
        ]

        technical_count = sum(1 for term in technical_terms if term in text.lower())

        if avg_sentence_length > 25 and technical_count > 5:
            return "experto"
        elif avg_sentence_length > 15 and technical_count > 2:
            return "intermedio"
        else:
            return "principiante"

    def save_scraped_data(
        self, content_list: List[ScrapedContent], filename: str
    ) -> None:
        """
        Save scraped content to CSV file

        Args:
            content_list: List of ScrapedContent objects
            filename: Output filename
        """
        data = []
        for content in content_list:
            data.append(
                {
                    "url": content.url,
                    "title": content.title,
                    "content": content.content,
                    "source": content.source,
                    "difficulty_level": content.difficulty_level,
                    "topic": content.topic,
                }
            )

        df = pd.DataFrame(data)
        output_path = os.path.join(data_config.scraped_data_dir, filename)
        df.to_csv(output_path, index=False, encoding="utf-8")

        logger.info(f"Saved {len(data)} items to {output_path}")

    def run_scraping_pipeline(self) -> None:
        """
        Run the complete scraping pipeline
        """
        logger.info("Starting scraping pipeline...")

        # Biology topics to scrape from Wikipedia
        biology_topics = [
            "Célula",
            "Homeostasis",
            "Fotosíntesis",
            "Respiración_celular",
            "ADN",
            "ARN",
            "Proteína",
            "Enzima",
            "Mitocondria",
            "Núcleo_celular",
            "Membrana_celular",
            "Sistema_nervioso",
            "Neurona",
            "Sinapsis",
            "Potencial_de_acción",
            "Sistema_cardiovascular",
            "Corazón",
            "Sangre",
            "Sistema_digestivo",
            "Metabolismo",
            "Genética",
            "Cromosoma",
            "Gen",
            "Mutación",
            "Evolución",
            "Selección_natural",
            "Ecosistema",
            "Biodiversidad",
            "Tejido_biológico",
            "Órgano",
        ]

        # Scrape Wikipedia content
        wikipedia_content = self.scrape_wikipedia_biology(biology_topics)
        if wikipedia_content:
            self.save_scraped_data(wikipedia_content, "wikipedia_biology.csv")

        # Educational URLs (these would need to be real educational sites)
        educational_urls = [
            # Add real educational URLs here
        ]

        if educational_urls:
            educational_content = self.scrape_educational_sites(educational_urls)
            if educational_content:
                self.save_scraped_data(educational_content, "educational_sites.csv")

        logger.info("Scraping pipeline completed!")


def main():
    """Main function for running the scraper"""
    scraper = WebScraper()
    scraper.run_scraping_pipeline()


if __name__ == "__main__":
    main()
