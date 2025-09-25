## Adding New Data Modalities

To add a new data modality (e.g., weather, land cover, etc.), follow these steps:

### 1. Create the Modality Function

Create a new file in `src/cropfm/data/modalities/` using the template structure:

Edit your new modality file following this structure:

```python
# src/cropfm/data/modalities/your_modality.py
import ee
import pandas as pd
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def your_modality(
    cfg: DictConfig,
    point: ee.Geometry.Point,
    **kwargs
) -> pd.DataFrame:
    """
    Extract your modality data for a point.
    
    Args:
        cfg: Hydra config containing modality parameters
        point: Geographic point to extract data from
        **kwargs: Additional parameters (can override config)
        
    Returns:
        pd.DataFrame: Extracted data with metadata
    """
    
    log.info(f"Starting {cfg.name} extraction")
    
    # Your Earth Engine data extraction logic here
    # Example:
    # collection = ee.ImageCollection('YOUR/GEE/COLLECTION')
    # image = collection.filterBounds(point).first()
    # sample = image.sample(region=point, scale=30, numPixels=1)
    # data = sample.getInfo()
    
    # Process and return as DataFrame
    df = pd.DataFrame(your_processed_data)
    
    log.info(f"Successfully extracted {len(df)} observations")
    return df
```

### 2. Add to Package Imports

Update `src/cropfm/data/modalities/__init__.py` to include your new modality:

```python
# src/cropfm/data/modalities/__init__.py
from .sentinel2 import sentinel2
from .fapar import fapar
from .soil import soil
from .elevation import elevation
from .your_modality import your_modality  # Add this line

__all__ = ['sentinel2', 'fapar', 'soil', 'elevation', 'your_modality']
```

### 3. Add to Download Script

Update `src/cropfm/data/download_modalities.py` to import your new modality:

```python
# src/cropfm/data/download_modalities.py
from cropfm.data.modalities import (
    sentinel2,
    fapar,
    soil,
    elevation,
    your_modality  # Add this line
)
```

### 4. Add Configuration

Add your modality configuration to `configs/data/modalities.yaml`:

```yaml
# Add your modality config
your_modality:
  log_level: *log_level
  name: your_modality
  # Add your specific parameters here
  collection: "YOUR/GEE/COLLECTION"
  bands:
    - band1
    - band2
  # Any other parameters your modality needs
  
# Add to the downloading list
modalities:
  - sentinel2
  - fapar
  - soil
  - elevation
  - your_modality  # Add this line

date_range: # only if needed
  start_date: *start_date 
  end_date: *end_date
```



### 5. Run the Complete Pipeline

Test with the full download pipeline:

```bash
python src/cropfm/data/download_modalities.py
```

## Configuration
All modalities are configured via `configs/data/modalities.yaml` using Hydra. You can:


## Logging

The project uses Python's logging module with configurable levels:

- `DEBUG`: Detailed extraction information
- `INFO`: Progress and summary information  

Set logging level in config.