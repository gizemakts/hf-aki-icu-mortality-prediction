# Third-party software notices

## MIT-LCP mimic-code — Glasgow Coma Scale (GCS) logic

The GCS component aggregation, previous-observation handling, and endotracheal verbal-response rules used in this repository adapt the MIT Laboratory for Computational Physiology's `mimic-code` implementation.

- Original source: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/gcs.sql
- Original license: https://github.com/MIT-LCP/mimic-code/blob/main/LICENSE
- Adapted locations: `scripts/preprocessing/predictors/predictor_inventory.py` and `scripts/modeling/harmonized/fit_harmonized_model.py`

The upstream copyright and permission notice is reproduced below in full.

MIT License

Copyright (c) 2019 MIT Laboratory for Computational Physiology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
