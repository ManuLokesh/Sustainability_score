from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
import json
import os

app = Flask(__name__)
CORS(app)

# --- Config ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///products.db'
db = SQLAlchemy(app)

# --- AI Configuration (Choose one or multiple) ---
# Option 1: OpenAI API
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key-here')

# Option 2: Hugging Face API (Free tier available)
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', 'your-hf-api-key-here')
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"

# Option 3: Google AI (Free tier available) 
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'your-google-ai-key-here')

# --- Ranges for Normalization ---
RANGES = {
    'GWP': {'min': 0, 'max': 100},
    'Cost': {'min': 0, 'max': 100},
    'Circularity': {'min': 0, 'max': 100},
}

def normalize(value, min_val, max_val, invert=False):
    if max_val == min_val:
        return 0.0
    norm = (value - min_val) / (max_val - min_val)
    if invert:
        norm = 1 - norm
    return max(0, min(1, norm))

def get_rating(score):
    """Convert score to rating (A, B, C, D)"""
    if score >= 80:
        return "A"
    elif score >= 65:
        return "B"
    elif score >= 50:
        return "C"
    else:
        return "D"

def generate_ai_suggestions_openai(product_data):
    """Generate suggestions using OpenAI API"""
    try:
        prompt = f"""
        You are a sustainability expert. Analyze this product data and provide exactly 3 specific, actionable sustainability improvement suggestions:

        Product: {product_data.get('product_name', 'Unknown')}
        Materials: {product_data.get('materials', [])}
        Weight: {product_data.get('weight_grams', 0)} grams
        Transport: {product_data.get('transport', 'unknown')}
        Packaging: {product_data.get('packaging', 'unknown')}
        Global Warming Potential: {product_data.get('gwp', 0)}/100
        Cost Impact: {product_data.get('cost', 0)}/100
        Circularity Score: {product_data.get('circularity', 0)}/100

        Provide exactly 3 bullet points with actionable suggestions. Keep each suggestion under 15 words.
        """

        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.7
        }
        
        response = request.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            suggestions_text = result['choices'][0]['message']['content'].strip()
            # Parse bullet points
            suggestions = [s.strip().lstrip('•-*').strip() for s in suggestions_text.split('\n') if s.strip()]
            return suggestions[:3]  # Return exactly 3
        else:
            raise Exception(f"OpenAI API error: {response.status_code}")
            
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return generate_fallback_suggestions(product_data)

def generate_ai_suggestions_huggingface(product_data):
    """Generate suggestions using Hugging Face API (Free tier)"""
    try:
        prompt = f"As a sustainability expert, give 3 specific suggestions to improve the environmental impact of this product: {product_data.get('product_name', 'product')} made from {product_data.get('materials', [])} with GWP:{product_data.get('gwp', 0)}, transported by {product_data.get('transport', 'unknown')}, packaged in {product_data.get('packaging', 'unknown')}."
        
        headers = {
            'Authorization': f'Bearer {HUGGINGFACE_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "do_sample": True
            }
        }
        
        response = request.post(HUGGINGFACE_API_URL, headers=headers, json=data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                suggestions_text = result[0].get('generated_text', '').replace(prompt, '').strip()
                # Extract suggestions from text
                suggestions = [s.strip() for s in suggestions_text.split('.') if s.strip()][:3]
                return suggestions if suggestions else generate_fallback_suggestions(product_data)
        
        raise Exception(f"Hugging Face API error: {response.status_code}")
        
    except Exception as e:
        print(f"Hugging Face API error: {e}")
        return generate_fallback_suggestions(product_data)

def generate_ai_suggestions_google(product_data):
    """Generate suggestions using Google AI API (Free tier)"""
    try:
        # Using Google's Generative AI (Gemini) - requires google-generativeai library
        import google.generativeai as genai
        
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        As a sustainability expert, analyze this product and provide exactly 3 specific improvement suggestions:
        
        Product: {product_data.get('product_name', 'Unknown')}
        Materials: {', '.join(product_data.get('materials', []))}
        Transport: {product_data.get('transport', 'unknown')}
        Packaging: {product_data.get('packaging', 'unknown')}
        Environmental Impact Score: {product_data.get('gwp', 0)}/100
        
        Format as 3 numbered suggestions, each under 15 words.
        """
        
        response = model.generate_content(prompt)
        suggestions_text = response.text.strip()
        
        # Parse numbered suggestions
        suggestions = []
        for line in suggestions_text.split('\n'):
            line = line.strip()
            if line and (line.startswith(('1.', '2.', '3.')) or line.startswith('-')):
                suggestion = line.split('.', 1)[-1].strip().lstrip('-').strip()
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions[:3] if suggestions else generate_fallback_suggestions(product_data)
        
    except Exception as e:
        print(f"Google AI API error: {e}")
        return generate_fallback_suggestions(product_data)

def generate_fallback_suggestions(product_data):
    """Fallback suggestions if AI APIs fail"""
    suggestions = []
    
    # Material-based suggestions
    materials = product_data.get('materials', [])
    if isinstance(materials, str):
        materials = materials.split(',')
    
    materials_lower = [m.lower().strip() for m in materials]
    
    if any('plastic' in m for m in materials_lower):
        suggestions.append("Replace plastic components with biodegradable alternatives")
    elif any('metal' in m or 'steel' in m or 'aluminum' in m for m in materials_lower):
        suggestions.append("Use recycled metal content to reduce mining impact")
    
    # Transport suggestions
    transport = product_data.get('transport', '').lower()
    if transport == 'air':
        suggestions.append("Switch from air to sea transport to reduce carbon emissions by 80%")
    elif transport == 'truck':
        suggestions.append("Consider rail transport for 75% lower emissions than trucking")
    
    # Packaging suggestions
    packaging = product_data.get('packaging', '').lower()
    if 'plastic' in packaging:
        suggestions.append("Use compostable packaging to eliminate plastic waste")
    elif 'non-recyclable' in packaging:
        suggestions.append("Switch to recyclable packaging materials")
    
    # Score-based suggestions
    gwp = float(product_data.get('gwp', 0))
    cost = float(product_data.get('cost', 0))
    circularity = float(product_data.get('circularity', 0))
    
    if gwp > 70 and len(suggestions) < 3:
        suggestions.append("Implement carbon offset programs for high-impact products")
    elif gwp > 50 and len(suggestions) < 3:
        suggestions.append("Source materials locally to reduce transportation emissions")
    
    if cost > 70 and len(suggestions) < 3:
        suggestions.append("Optimize production efficiency to reduce resource consumption")
    
    if circularity < 30 and len(suggestions) < 3:
        suggestions.append("Design for disassembly to enable component recycling")
    elif circularity < 50 and len(suggestions) < 3:
        suggestions.append("Implement take-back program for end-of-life processing")
    
    # Weight-based suggestions
    weight = product_data.get('weight_grams', 0)
    if weight > 1000 and len(suggestions) < 3:
        suggestions.append("Use lightweight materials to reduce shipping emissions")
    
    # Ensure we have at least 3 suggestions
    default_suggestions = [
        "Conduct lifecycle assessment to identify improvement opportunities",
        "Source renewable energy for manufacturing processes",
        "Implement supplier sustainability certification requirements"
    ]
    
    while len(suggestions) < 3:
        for default in default_suggestions:
            if default not in suggestions:
                suggestions.append(default)
                break
        if len(suggestions) >= 3:
            break
    
    return suggestions[:3]

def generate_ai_suggestions(product_data):
    """Main function to generate AI suggestions - tries multiple APIs"""
    
    # Try OpenAI first (most reliable)
    if OPENAI_API_KEY and OPENAI_API_KEY != 'your-openai-api-key-here':
        try:
            return generate_ai_suggestions_openai(product_data)
        except Exception as e:
            print(f"OpenAI failed: {e}")
    
    # Try Hugging Face (free tier)
    if HUGGINGFACE_API_KEY and HUGGINGFACE_API_KEY != 'your-hf-api-key-here':
        try:
            return generate_ai_suggestions_huggingface(product_data)
        except Exception as e:
            print(f"Hugging Face failed: {e}")
    
    # Try Google AI (free tier)
    if GOOGLE_API_KEY and GOOGLE_API_KEY != 'your-google-ai-key-here':
        try:
            return generate_ai_suggestions_google(product_data)
        except Exception as e:
            print(f"Google AI failed: {e}")
    
    # Fallback to rule-based suggestions
    print("Using fallback suggestions - no AI API configured")
    return generate_fallback_suggestions(product_data)

# --- Enhanced Product Model ---
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    materials = db.Column(db.String(200))
    weight_grams = db.Column(db.Float)
    transport = db.Column(db.String(50))
    packaging = db.Column(db.String(100))
    gwp = db.Column(db.Float)
    cost = db.Column(db.Float)
    circularity = db.Column(db.Float)
    weight_gwp = db.Column(db.Float)
    weight_cost = db.Column(db.Float)
    weight_circularity = db.Column(db.Float)
    norm_gwp = db.Column(db.Float)
    norm_cost = db.Column(db.Float)
    norm_circularity = db.Column(db.Float)
    final_score = db.Column(db.Float)
    rating = db.Column(db.String(1))
    suggestions = db.Column(db.Text)  # Store as JSON string
    ai_used = db.Column(db.String(20))  # Track which AI was used
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

@app.route("/")
def home():
    return render_template("index.html")

# --- CASE STUDY ENDPOINT: POST /score ---
@app.route("/score", methods=["POST"])
def calculate_and_store_score():
    try:
        data = request.json
        
        # Input validation
        required_fields = ['product_name', 'gwp', 'cost', 'circularity']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        name = data.get('product_name', 'Unnamed')
        materials = data.get('materials', [])
        if isinstance(materials, str):
            materials = materials.split(',')
        materials_str = ",".join([m.strip() for m in materials]) if materials else ""
        
        weight_grams = float(data.get('weight_grams', 0))
        transport = data.get('transport', 'unknown')
        packaging = data.get('packaging', 'unknown')
        gwp = float(data['gwp'])
        cost = float(data['cost'])
        circularity = float(data['circularity'])
        
        weights = data.get('weights', {
            'gwp': 0.4,
            'cost': 0.3,
            'circularity': 0.3
        })

        # Validate weight sum
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            return jsonify({'error': 'Weights must sum to 1.0'}), 400

        # Normalize
        S_gwp = normalize(gwp, RANGES['GWP']['min'], RANGES['GWP']['max'], invert=True)
        S_cost = normalize(cost, RANGES['Cost']['min'], RANGES['Cost']['max'], invert=True)
        S_circ = normalize(circularity, RANGES['Circularity']['min'], RANGES['Circularity']['max'])

        final_score = (
            weights['gwp'] * S_gwp +
            weights['cost'] * S_cost +
            weights['circularity'] * S_circ
        ) * 100

        # Get rating and AI suggestions
        rating = get_rating(final_score)
        
        # *** REAL AI INTEGRATION HERE ***
        suggestions = generate_ai_suggestions(data)
        ai_used = "OpenAI" if OPENAI_API_KEY != 'your-openai-api-key-here' else "Fallback"

        # Save to DB
        product = Product(
            name=name,
            materials=materials_str,
            weight_grams=weight_grams,
            transport=transport,
            packaging=packaging,
            gwp=gwp,
            cost=cost,
            circularity=circularity,
            weight_gwp=weights['gwp'],
            weight_cost=weights['cost'],
            weight_circularity=weights['circularity'],
            norm_gwp=round(S_gwp, 3),
            norm_cost=round(S_cost, 3),
            norm_circularity=round(S_circ, 3),
            final_score=round(final_score, 2),
            rating=rating,
            suggestions=json.dumps(suggestions),  # Store as JSON
            ai_used=ai_used
        )

        db.session.add(product)
        db.session.commit()

        # Return response in case study format
        return jsonify({
            'product_name': name,
            'sustainability_score': round(final_score, 2),
            'rating': rating,
            'suggestions': suggestions
        })

    except Exception as e:
        print(f"Error in calculate_and_store_score: {e}")
        return jsonify({'error': str(e)}), 400

# --- CASE STUDY ENDPOINT: GET /history ---
@app.route("/history", methods=["GET"])
def get_all_products():
    try:
        # Get recent submissions (last 20)
        products = Product.query.order_by(Product.created_at.desc()).limit(20).all()
        
        result = []
        for p in products:
            suggestions = []
            if p.suggestions:
                try:
                    suggestions = json.loads(p.suggestions)
                except:
                    suggestions = []
            
            result.append({
                'id': p.id,
                'product_name': p.name,
                'materials': p.materials.split(",") if p.materials else [],
                'weight_grams': p.weight_grams,
                'transport': p.transport,
                'packaging': p.packaging,
                'sustainability_score': p.final_score,
                'rating': p.rating,
                'suggestions': suggestions,
                'ai_used': p.ai_used,
                'created_at': p.created_at.isoformat() if p.created_at else None
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- CASE STUDY ENDPOINT: GET /score-summary ---
@app.route("/score-summary", methods=["GET"])
def get_score_summary():
    try:
        products = Product.query.all()
        
        if not products:
            return jsonify({
                'total_products': 0,
                'average_score': 0,
                'ratings': {'A': 0, 'B': 0, 'C': 0, 'D': 0},
                'top_issues': [],
                'ai_usage': {}
            })
        
        total_products = len(products)
        average_score = sum(p.final_score for p in products) / total_products
        
        # Count ratings
        ratings = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        ai_usage = {}
        
        for p in products:
            if p.rating in ratings:
                ratings[p.rating] += 1
            
            # Track AI usage
            ai_type = p.ai_used or 'Unknown'
            ai_usage[ai_type] = ai_usage.get(ai_type, 0) + 1
        
        # Analyze top issues from AI suggestions
        all_suggestions = []
        for p in products:
            if p.suggestions:
                try:
                    suggestions = json.loads(p.suggestions)
                    all_suggestions.extend(suggestions)
                except:
                    pass
        
        # Extract common themes
        issues = []
        suggestion_text = ' '.join(all_suggestions).lower()
        if 'plastic' in suggestion_text:
            issues.append("Plastic material usage")
        if 'transport' in suggestion_text or 'shipping' in suggestion_text:
            issues.append("Transportation emissions")
        if 'packaging' in suggestion_text:
            issues.append("Packaging sustainability")
        if 'carbon' in suggestion_text or 'emissions' in suggestion_text:
            issues.append("Carbon footprint reduction")
        if 'recycl' in suggestion_text:
            issues.append("Recyclability improvements")
        if 'energy' in suggestion_text:
            issues.append("Energy efficiency")
        
        return jsonify({
            'total_products': total_products,
            'average_score': round(average_score, 1),
            'ratings': ratings,
            'top_issues': issues[:3],
            'ai_usage': ai_usage
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- UTILITY ENDPOINT: Test AI Connection ---
@app.route("/test-ai", methods=["GET"])
def test_ai_connection():
    """Test endpoint to check AI API connectivity"""
    test_data = {
        'product_name': 'Test Product',
        'materials': ['plastic', 'steel'],
        'transport': 'truck',
        'packaging': 'plastic',
        'gwp': 60,
        'cost': 40,
        'circularity': 30
    }
    
    suggestions = generate_ai_suggestions(test_data)
    
    return jsonify({
        'status': 'success',
        'suggestions': suggestions,
        'openai_configured': OPENAI_API_KEY != 'your-openai-api-key-here',
        'huggingface_configured': HUGGINGFACE_API_KEY != 'your-hf-api-key-here',
        'google_configured': GOOGLE_API_KEY != 'your-google-ai-key-here'
    })

# --- LEGACY ENDPOINTS (for backward compatibility) ---
@app.route("/api/score", methods=["POST"])
def legacy_calculate_score():
    return calculate_and_store_score()

@app.route("/api/products", methods=["GET"])
def legacy_get_products():
    return get_all_products()

@app.route("/api/update_weights", methods=["POST"])
def update_all_weights():
    try:
        weights = request.json.get("weights", {})
        if abs(sum(weights.values()) - 1.0) > 0.01:
            return jsonify({"error": "Weights must sum to 1.0"}), 400

        products = Product.query.all()
        for p in products:
            # Recalculate normalized scores
            S_gwp = normalize(p.gwp, RANGES['GWP']['min'], RANGES['GWP']['max'], invert=True)
            S_cost = normalize(p.cost, RANGES['Cost']['min'], RANGES['Cost']['max'], invert=True)
            S_circ = normalize(p.circularity, RANGES['Circularity']['min'], RANGES['Circularity']['max'])

            final_score = (
                weights['gwp'] * S_gwp +
                weights['cost'] * S_cost +
                weights['circularity'] * S_circ
            ) * 100

            # Update
            p.final_score = round(final_score, 2)
            p.rating = get_rating(final_score)
            p.weight_gwp = weights['gwp']
            p.weight_cost = weights['cost']
            p.weight_circularity = weights['circularity']

        db.session.commit()
        return jsonify({'message': 'Scores updated successfully'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
