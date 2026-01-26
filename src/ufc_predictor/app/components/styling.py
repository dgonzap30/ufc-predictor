"""
Custom CSS styling for UFC Sniper Dashboard
"""


def get_custom_css():
    """Return custom CSS for enhanced UI styling"""
    return """
    <style>
    /* Main container improvements */
    .main {
        padding: 1rem 2rem;
    }

    /* Card-like containers */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }

    /* Sniper bet success cards */
    .stAlert[data-baseweb="notification"][kind="success"] {
        border-left-color: #00C851;
        background-color: rgba(0, 200, 81, 0.1);
    }

    /* Pass warning cards */
    .stAlert[data-baseweb="notification"][kind="warning"] {
        border-left-color: #FFB300;
        background-color: rgba(255, 179, 0, 0.1);
    }

    /* Error/High Risk cards */
    .stAlert[data-baseweb="notification"][kind="error"] {
        border-left-color: #E31837;
        background-color: rgba(227, 24, 55, 0.1);
    }

    /* Info cards */
    .stAlert[data-baseweb="notification"][kind="info"] {
        border-left-color: #2196F3;
        background-color: rgba(33, 150, 243, 0.1);
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.8;
    }

    /* Buttons */
    .stButton button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(227, 24, 55, 0.3);
    }

    /* Primary button */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #E31837 0%, #C41230 100%);
        border: none;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid rgba(227, 24, 55, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 8px 8px 0 0;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(227, 24, 55, 0.1);
        border-bottom: 3px solid #E31837;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.05rem;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        background-color: rgba(255, 255, 255, 0.02);
        transition: all 0.2s ease;
    }

    .streamlit-expanderHeader:hover {
        background-color: rgba(227, 24, 55, 0.1);
    }

    /* Sniper bet indicator */
    .sniper-bet-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00C851 0%, #007E33 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-left: 8px;
        letter-spacing: 0.5px;
    }

    /* Pass indicator */
    .pass-badge {
        display: inline-block;
        background: linear-gradient(135deg, #FFB300 0%, #E69800 100%);
        color: #000;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-left: 8px;
        letter-spacing: 0.5px;
    }

    /* Sidebar improvements */
    [data-testid="stSidebar"] {
        background-color: #0A0A0A;
        border-right: 1px solid rgba(227, 24, 55, 0.2);
    }

    /* Dividers */
    hr {
        margin: 0.5rem 0;
        border-color: rgba(227, 24, 55, 0.2);
    }

    /* Table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }

    .dataframe thead tr th {
        background-color: rgba(227, 24, 55, 0.2) !important;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }

    .dataframe tbody tr:hover {
        background-color: rgba(227, 24, 55, 0.05);
    }

    /* Headers */
    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #E31837 0%, #FFB300 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #E31837 !important;
    }

    /* Tooltips */
    [data-testid="stTooltipIcon"] {
        color: #E31837;
    }

    /* Input fields */
    input, textarea, select {
        border-radius: 6px !important;
    }

    /* Number input buttons */
    button[kind="stepUp"], button[kind="stepDown"] {
        background-color: rgba(227, 24, 55, 0.1) !important;
    }

    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 0.75rem 0;
    }

    .stat-card {
        background: linear-gradient(135deg, rgba(227, 24, 55, 0.1) 0%, rgba(227, 24, 55, 0.05) 100%);
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #E31837;
    }

    .stat-label {
        text-transform: uppercase;
        font-size: 0.75rem;
        opacity: 0.7;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }

    .stat-value {
        font-size: 2rem;
        font-weight: 700;
    }
    </style>
    """
