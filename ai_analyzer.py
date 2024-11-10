import google.generativeai as genai
import os

def get_stock_recommendation(data, stock_symbol, language="Turkish"):
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel('gemini-pro')
    
    # Update language templates to include full analysis template in each language
    language_templates = {
        "Turkish": '''
            Analiz et ve Türkçe yanıt ver:
            Hisse: {stock_symbol}
            
            Teknik Göstergeler:
            - RSI: {rsi:.2f}
            - MACD: {macd:.2f}
            - Stokastik: {stochastic:.2f}
            
            Model Tahminleri:
            {model_predictions}
            
            Metrikler:
            - Aylık Getiri: {monthly_returns:.2%}
            - Volatilite: {volatility:.2%}
            - Sharpe Oranı: {sharpe_ratio:.2f}
            - Beta: {beta}
            - Alpha: {alpha}
            - Bilgi Oranı: {info_ratio}
            
            Lütfen analiz et:
            1. Model tahminlerinin uyumu
            2. Teknik gösterge sinyalleri
            3. Risk seviyesi
            4. Potansiyel kar/zarar senaryoları
            5. Piyasa duyarlılığı
            6. Sektör performans etkisi
            
            Sağla:
            - AL/TUT/SAT tavsiyesi
            - Güven seviyesi
            - Detaylı gerekçelendirme
            - Risk değerlendirmesi
            - Fiyat hedefleri
        ''',
        "English": '''
            Analyze and respond in English:
            Stock: {stock_symbol}
            
            Technical Indicators:
            - RSI: {rsi:.2f}
            - MACD: {macd:.2f}
            - Stochastic: {stochastic:.2f}
            
            Model Predictions:
            {model_predictions}
            
            Metrics:
            - Monthly Returns: {monthly_returns:.2%}
            - Volatility: {volatility:.2%}
            - Sharpe Ratio: {sharpe_ratio:.2f}
            - Beta: {beta}
            - Alpha: {alpha}
            - Information Ratio: {info_ratio}
            
            Please analyze:
            1. Model prediction consensus
            2. Technical indicator signals
            3. Risk level
            4. Potential profit/loss scenarios
            5. Market sentiment
            6. Sector performance impact
            
            Provide:
            - BUY/HOLD/SELL recommendation
            - Confidence level
            - Detailed reasoning
            - Risk assessment
            - Price targets
        ''',
        "German": '''
            Analysieren Sie und antworten Sie auf Deutsch:
            Aktie: {stock_symbol}
            
            Technische Indikatoren:
            - RSI: {rsi:.2f}
            - MACD: {macd:.2f}
            - Stochastik: {stochastic:.2f}
            
            Modellvorhersagen:
            {model_predictions}
            
            Kennzahlen:
            - Monatliche Rendite: {monthly_returns:.2%}
            - Volatilität: {volatility:.2%}
            - Sharpe Ratio: {sharpe_ratio:.2f}
            - Beta: {beta}
            - Alpha: {alpha}
            - Informationsrate: {info_ratio}
            
            Bitte analysieren Sie:
            1. Konsens der Modellvorhersagen
            2. Signale der technischen Indikatoren
            3. Risikoniveau
            4. Potenzielle Gewinn-/Verlustszenarien
            5. Marktstimmung
            6. Sektorperformance-Auswirkung
            
            Bereitstellen:
            - KAUFEN/HALTEN/VERKAUFEN Empfehlung
            - Konfidenzniveau
            - Detaillierte Begründung
            - Risikobewertung
            - Kursziele
        ''',
        "Russian": '''
            Проанализируйте и ответьте на русском:
            Акция: {stock_symbol}
            
            Технические индикаторы:
            - RSI: {rsi:.2f}
            - MACD: {macd:.2f}
            - Стохастик: {stochastic:.2f}
            
            Прогнозы моделей:
            {model_predictions}
            
            Метрики:
            - Месячная доходность: {monthly_returns:.2%}
            - Волатильность: {volatility:.2%}
            - Коэффициент Шарпа: {sharpe_ratio:.2f}
            - Бета: {beta}
            - Альфа: {alpha}
            - Коэффициент информации: {info_ratio}
            
            Пожалуйста, проанализируйте:
            1. Консенсус прогнозов моделей
            2. Сигналы технических индикаторов
            3. Уровень риска
            4. Потенциальные сценарии прибыли/убытков
            5. Настроение рынка
            6. Влияние показателей сектора
            
            Предоставьте:
            - Рекомендация ПОКУПАТЬ/ДЕРЖАТЬ/ПРОДАВАТЬ
            - Уровень уверенности
            - Детальное обоснование
            - Оценка рисков
            - Целевые цены
        ''',
        "French": '''
            Analysez et répondez en français:
            Action: {stock_symbol}
            
            Indicateurs techniques:
            - RSI: {rsi:.2f}
            - MACD: {macd:.2f}
            - Stochastique: {stochastic:.2f}
            
            Prédictions des modèles:
            {model_predictions}
            
            Métriques:
            - Rendements mensuels: {monthly_returns:.2%}
            - Volatilité: {volatility:.2%}
            - Ratio de Sharpe: {sharpe_ratio:.2f}
            - Bêta: {beta}
            - Alpha: {alpha}
            - Ratio d'information: {info_ratio}
            
            Veuillez analyser:
            1. Consensus des prédictions des modèles
            2. Signaux des indicateurs techniques
            3. Niveau de risque
            4. Scénarios potentiels de profit/perte
            5. Sentiment du marché
            6. Impact de la performance sectorielle
            
            Fournir:
            - Recommandation ACHETER/CONSERVER/VENDRE
            - Niveau de confiance
            - Raisonnement détaillé
            - Évaluation des risques
            - Objectifs de prix
        ''',
        "Spanish": '''
            Analiza y responde en español:
            Acción: {stock_symbol}
            
            Indicadores técnicos:
            - RSI: {rsi:.2f}
            - MACD: {macd:.2f}
            - Estocástico: {stochastic:.2f}
            
            Predicciones de modelos:
            {model_predictions}
            
            Métricas:
            - Rendimientos mensuales: {monthly_returns:.2%}
            - Volatilidad: {volatility:.2%}
            - Ratio de Sharpe: {sharpe_ratio:.2f}
            - Beta: {beta}
            - Alpha: {alpha}
            - Ratio de información: {info_ratio}
            
            Por favor analiza:
            1. Consenso de predicciones de modelos
            2. Señales de indicadores técnicos
            3. Nivel de riesgo
            4. Escenarios potenciales de ganancia/pérdida
            5. Sentimiento del mercado
            6. Impacto del rendimiento sectorial
            
            Proporcionar:
            - Recomendación COMPRAR/MANTENER/VENDER
            - Nivel de confianza
            - Razonamiento detallado
            - Evaluación de riesgos
            - Objetivos de precio
        '''
    }
    
    # Add error handling for language selection
    if language not in language_templates:
        language = "English"  # Default fallback
    
    # Format the template with the data
    prompt = language_templates[language].format(
        stock_symbol=stock_symbol,
        rsi=data['technical_indicators']['rsi'],
        macd=data['technical_indicators']['macd'],
        stochastic=data['technical_indicators']['stochastic'],
        model_predictions=data['model_predictions'],
        monthly_returns=data['metrics']['Monthly Returns'],
        volatility=data['metrics']['Volatility'],
        sharpe_ratio=data['metrics']['Sharpe Ratio'],
        beta=data['metrics'].get('Beta', 'N/A'),
        alpha=data['metrics'].get('Alpha', 'N/A'),
        info_ratio=data['metrics'].get('Information Ratio', 'N/A')
    )
    
    response = model.generate_content(prompt)
    return response.text
