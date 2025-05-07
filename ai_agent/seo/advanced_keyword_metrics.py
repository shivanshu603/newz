from typing import Dict, List
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from collections import defaultdict
from datetime import datetime, timedelta
import torch
from transformers import pipeline
from utils import logger
from urllib.parse import quote

class AdvancedKeywordMetrics:
    def __init__(self):
        self.embedding_model = pipeline('feature-extraction', model='sentence-transformers/all-MiniLM-L6-v2')
        self.scaler = StandardScaler()
        self.difficulty_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    async def calculate_comprehensive_metrics(self, keyword: str) -> Dict:
        """Calculate comprehensive keyword metrics using ML"""
        try:
            # Gather real-time data
            serp_data = await self._get_realtime_serp_data(keyword)
            trend_data = await self._get_trend_metrics(keyword)
            competition_data = await self._analyze_competition(keyword)
            
            # Calculate base metrics
            metrics = {
                'search_metrics': await self._calculate_search_metrics(keyword),
                'content_metrics': self._analyze_content_requirements(serp_data),
                'competition_metrics': competition_data,
                'trend_metrics': trend_data,
                'intent_metrics': self._analyze_search_intent(keyword, serp_data)
            }
            
            # Apply ML scoring
            final_scores = self._calculate_ml_scores(metrics)
            
            # Generate forecasts
            forecasts = self._generate_forecasts(metrics)
            
            return {
                'scores': final_scores,
                'metrics': metrics,
                'forecasts': forecasts,
                'opportunities': self._identify_opportunities(metrics),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}

    def _calculate_ml_scores(self, metrics: Dict) -> Dict:
        """Calculate scores using ML models"""
        try:
            # Prepare feature vector
            features = np.array([
                metrics['search_metrics']['volume'],
                metrics['competition_metrics']['difficulty'],
                metrics['trend_metrics']['momentum'],
                metrics['content_metrics']['quality_required'],
                metrics['intent_metrics']['commercial_intent']
            ]).reshape(1, -1)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Calculate scores
            return {
                'overall_score': float(self.difficulty_model.predict(scaled_features)[0]),
                'potential_score': self._calculate_potential(scaled_features),
                'confidence_score': self._calculate_confidence(metrics)
            }
            
        except Exception as e:
            logger.error(f"Error in ML scoring: {e}")
            return {}

    def _identify_opportunities(self, metrics: Dict) -> Dict:
        """Identify keyword opportunities using ML insights"""
        opportunities = defaultdict(list)
        
        try:
            # Content opportunities
            if metrics['content_metrics']['gaps']:
                opportunities['content'].extend(metrics['content_metrics']['gaps'])
            
            # Competition opportunities
            if metrics['competition_metrics']['difficulty'] < 0.6:
                opportunities['competition'].append({
                    'type': 'ranking_potential',
                    'description': 'Moderate competition level indicates ranking potential',
                    'score': 1 - metrics['competition_metrics']['difficulty']
                })
            
            # Trend opportunities
            if metrics['trend_metrics']['momentum'] > 0:
                opportunities['trends'].append({
                    'type': 'growing_trend',
                    'description': 'Positive trend momentum indicates growth potential',
                    'score': metrics['trend_metrics']['momentum']
                })
                
            return dict(opportunities)
            
        except Exception as e:
            logger.error(f"Error identifying opportunities: {e}")
            return {}

    def _generate_forecasts(self, metrics: Dict) -> Dict:
        """Generate keyword trend forecasts"""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            # Prepare time series data
            historical_data = np.array(metrics['trend_metrics']['historical_data'])
            
            # Fit model
            model = ExponentialSmoothing(
                historical_data,
                seasonal_periods=12,
                trend='add',
                seasonal='add'
            ).fit()
            
            # Generate forecasts
            forecast_periods = 6  # 6 months
            forecasts = model.forecast(forecast_periods)
            
            return {
                'values': forecasts.tolist(),
                'confidence_intervals': self._calculate_confidence_intervals(forecasts),
                'seasonality': self._detect_seasonality(historical_data)
            }
            
        except Exception as e:
            logger.error(f"Error generating forecasts: {e}")
            return {}

    def _calculate_confidence_intervals(self, forecasts: np.ndarray) -> Dict:
        """Calculate confidence intervals for forecasts"""
        try:
            return {
                '95%': {
                    'lower': (forecasts - 2 * forecasts.std()).tolist(),
                    'upper': (forecasts + 2 * forecasts.std()).tolist()
                },
                '80%': {
                    'lower': (forecasts - 1.28 * forecasts.std()).tolist(),
                    'upper': (forecasts + 1.28 * forecasts.std()).tolist()
                }
            }
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {e}")
            return {}

    async def _get_realtime_serp_data(self, keyword: str) -> Dict:
        """Get SERP data with improved error handling"""
        retries = 3
        for attempt in range(retries):
            try:
                # Add delay between retries
                if attempt > 0:
                    await asyncio.sleep(2 * attempt)
                    
                encoded_query = quote(keyword.encode('utf-8').decode())
                url = f"https://www.google.com/search?q={encoded_query}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5'
                }
                
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            return {
                                'titles': [t.text for t in soup.select('.r')],
                                'snippets': [s.text for s in soup.select('.s')],
                                'total_results': self._extract_result_count(soup),
                                'features': self._extract_serp_features(soup)
                            }
                            
            except (aiohttp.ServerDisconnectedError, aiohttp.ClientError) as e:
                if attempt == retries - 1:
                    logger.error(f"Final retry failed for SERP data: {e}")
                    return {}
                logger.warning(f"Retry {attempt + 1} for SERP data after error: {e}")
            except Exception as e:
                logger.error(f"Error getting SERP data: {e}")
                return {}
                
        return {}

    async def _get_trend_metrics(self, keyword: str) -> Dict:
        """Get trend metrics via scraping instead of APIs"""
        try:
            # Scrape Google Trends
            url = f"https://trends.google.com/trends/explore?q={quote(keyword)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract trend data from page
                        trend_data = self._extract_trend_data(soup)
                        return {
                            'trend_score': self._calculate_trend_score(trend_data),
                            'historical_data': trend_data.get('history', []),
                            'momentum': self._calculate_momentum(trend_data)
                        }
            
            return {'trend_score': 0.5, 'historical_data': [], 'momentum': 0}
            
        except Exception as e:
            logger.error(f"Error getting trend metrics: {e}")
            return {'trend_score': 0.5, 'historical_data': [], 'momentum': 0}
