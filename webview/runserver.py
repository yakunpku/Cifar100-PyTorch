
import config as cfg
from app import app

app.run(host='0.0.0.0', port=cfg.port, debug=True)
