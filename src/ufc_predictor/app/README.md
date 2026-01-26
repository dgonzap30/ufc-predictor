# UFC Sniper Dashboard

Modern, multi-page Streamlit application for UFC fight predictions and betting analysis.

## Structure

```
app/
├── Home.py                    # Main dashboard page (run this with streamlit)
├── pages/
│   ├── 1_Manual_Matchup.py   # Custom fighter vs fighter predictions
│   └── 2_Strategy_Guide.py   # Educational content & strategy breakdown
├── components/
│   ├── styling.py            # Custom CSS styling
│   └── cards.py              # Reusable UI components
└── dashboard_old.py          # Legacy single-page dashboard (backup)
```

## Running the Dashboard

### Option 1: Using the launch script (recommended)
```bash
./scripts/run_dashboard.sh
```

### Option 2: Direct Streamlit command
```bash
streamlit run src/ufc_predictor/app/Home.py
```

## Features

### Home Page
- **Event Loading**: Load upcoming or recent UFC events
- **Quick Stats**: Summary metrics for loaded events
- **Sniper Summary**: Table of recommended bets
- **All Fights Tab**: View all predictions with expandable cards
- **Sniper Bets Only Tab**: Filter to only profitable recommendations

### Manual Matchup Page
- Create custom fighter vs fighter predictions
- Optional odds input for EV calculation
- Detailed betting analysis breakdown
- Sniper criteria validation

### Strategy Guide Page
- Comprehensive explanation of Sniper Strategy
- Performance breakdown by weight class
- Model architecture details
- Betting guidelines and best practices
- FAQ section

## Design Features

### Modern UI/UX
- **Dark Theme**: UFC-inspired color scheme (black/red/gold)
- **Custom CSS**: Enhanced styling for cards, buttons, metrics
- **Responsive Layout**: Works on desktop and mobile
- **Interactive Components**: Expandable cards, sortable tables
- **Visual Indicators**: Color-coded confidence levels, status badges

### Removed
- ❌ Static heatmap image (replaced with interactive Strategy Guide)
- ❌ Single-page monolithic structure
- ❌ Ugly default Streamlit styling

### Improved
- ✅ Multi-page navigation
- ✅ Modular, reusable components
- ✅ Better information hierarchy
- ✅ Clearer call-to-actions
- ✅ Enhanced visual feedback

## Configuration

Theme configuration is stored in `.streamlit/config.toml`:
- Primary color: UFC Red (#E31837)
- Background: Deep Black (#0E0E0E)
- Dark mode by default

## Development

### Adding New Pages
Create a new file in `pages/` with the naming convention:
```
N_Page_Name.py
```

Where N is the page number (determines order in sidebar).

### Creating Components
Add reusable UI functions to `components/` and import in pages:
```python
from ufc_predictor.app.components.cards import render_fight_card
```

### Customizing Styles
Edit `components/styling.py` to modify the CSS.

## Migration Notes

The old single-page dashboard has been backed up as `dashboard_old.py`. The new multi-page
structure provides better organization, navigation, and maintainability.

Key improvements:
- Separation of concerns (event analysis vs manual predictions vs education)
- Reusable components reduce code duplication
- Custom styling creates consistent brand identity
- Better UX with clear information architecture
