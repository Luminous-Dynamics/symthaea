# WebGL Testing Solution for Terra Atlas Globe

## The Problem
WebGL-based 3D visualizations (like the Terra Atlas globe) cannot be tested with automated screenshots because:
1. **WebGL requires GPU** - Headless browsers don't have GPU access
2. **NixOS limitations** - Browser dependencies difficult in headless mode
3. **Playwright limitation** - Can't render 3D canvas elements without display

## The Solution: Hybrid Testing Approach

### ✅ What We **CAN** Automate (90%)
Using WebPilot to test everything except visual rendering:

```python
# Test component loads
assert browser.element_exists("canvas")  # WebGL canvas exists

# Test data fetches correctly
assert browser.count_elements(".project-marker") > 0

# Test interactions work
browser.click("button[data-filter='solar']")
assert "filter=solar" in browser.get_url()

# Test performance
load_time = measure_page_load()
assert load_time < 5.0  # seconds
```

### ⚠️ What Requires Manual Verification (10%)
Visual appearance only (when it actually changes):
- Globe renders correctly
- Textures look good
- Colors are accurate
- Animations are smooth
- Markers in right positions

## Quick Implementation Guide

### 1. Create Test Files

**tests/webpilot-globe-test.py**
```python
#!/usr/bin/env python3
"""Terra Atlas Globe WebPilot Tests"""
import sys
sys.path.insert(0, '/srv/luminous-dynamics/_development/web-automation/claude-webpilot/src')

from webpilot.core import PlaywrightAutomation

def test_globe_loads():
    with PlaywrightAutomation(headless=False) as browser:
        browser.navigate("http://localhost:3000")

        # Test 1: Canvas exists
        canvas = browser.page.query_selector("canvas")
        assert canvas, "Globe canvas should exist"

        # Test 2: No errors
        errors = browser.page.evaluate("() => window.__ERRORS__ || []")
        assert len(errors) == 0, "Should have no console errors"

        # Test 3: Data loaded
        markers = browser.page.query_selector_all(".project-marker")
        assert len(markers) > 0, "Should have project markers"

        print("✅ All automated tests passed!")

if __name__ == "__main__":
    test_globe_loads()
```

### 2. Create Manual Verification Script

**scripts/visual-verify-globe.sh**
```bash
#!/usr/bin/env bash
echo "🌍 Terra Atlas Visual Verification"
echo "==================================="
echo ""

# Start dev server
npm run dev &
DEV_PID=$!
sleep 3

echo "✅ Dev server at http://localhost:3000"
echo ""
echo "📋 Manual Checklist:"
echo "  [ ] Globe renders correctly"
echo "  [ ] Markers visible"
echo "  [ ] Interactions work"
echo "  [ ] Animations smooth"
echo ""
echo "Press ENTER when verified..."
read

kill $DEV_PID
echo "✅ Verification complete"
```

### 3. Add to package.json

```json
{
  "scripts": {
    "test:auto": "python tests/webpilot-globe-test.py",
    "test:visual": "./scripts/visual-verify-globe.sh",
    "test:all": "npm run test:auto && npm run test:visual"
  }
}
```

## Usage Examples

### For Small Changes (CSS, colors)
```bash
# 1. Run automated tests
python tests/webpilot-globe-test.py

# 2. Quick manual check (30 seconds)
# Open http://localhost:3000 and verify

# 3. Commit
git commit -m "Update marker colors - verified manually"
```

### For Feature Changes
```bash
# 1. Write/update tests
# Edit tests/webpilot-globe-test.py

# 2. Run tests
python tests/webpilot-globe-test.py

# 3. Manual verification with checklist
./scripts/visual-verify-globe.sh

# 4. Take screenshots
# Save to docs/visual-evidence/

# 5. Commit with evidence
git commit -m "Add filter animations - tests + screenshots"
```

## WebPilot Capabilities for Globe Testing

### ✅ Can Test
- Component structure exists
- Data loads correctly
- State changes work
- URL routing works
- Event handlers fire
- Performance metrics
- Responsive breakpoints
- No console errors
- Memory leaks (over time)
- Accessibility (ARIA, keyboard nav)

### ❌ Cannot Test
- Actual visual appearance
- 3D rendering quality
- Animation smoothness (visual)
- Color/texture accuracy
- WebGL shader output

## Performance Comparison

| Approach | Time per Test | Coverage | False Negatives |
|----------|---------------|----------|-----------------|
| **Manual only** | 5-10 min | 50% | High |
| **Headless screenshots** | N/A | 0% | Doesn't work |
| **Hybrid (WebPilot + Manual)** | 30 sec + 2 min | 90% + 10% | Low |

**Hybrid approach = Best of both worlds**

## Example: Complete Test Suite

```python
# tests/complete-globe-test.py

def test_1_component_loads():
    """Test globe component structure exists"""
    with PlaywrightAutomation() as browser:
        browser.navigate("http://localhost:3000")
        assert browser.element_exists("canvas")
        assert browser.element_exists(".globe-container")
        print("✅ Component loads")

def test_2_data_fetches():
    """Test project data fetches and displays"""
    with PlaywrightAutomation() as browser:
        browser.navigate("http://localhost:3000")
        browser.wait(3000)  # Wait for data
        count = browser.count_elements(".project-marker")
        assert count > 0, f"Expected markers, got {count}"
        print(f"✅ Data loaded ({count} markers)")

def test_3_filters_work():
    """Test project type filters"""
    with PlaywrightAutomation() as browser:
        browser.navigate("http://localhost:3000")

        # Click solar filter
        browser.click("button:has-text('Solar')")
        browser.wait(1000)

        # Verify URL changed
        assert "filter=solar" in browser.get_url()
        print("✅ Filters work")

def test_4_responsive():
    """Test responsive layout"""
    with PlaywrightAutomation() as browser:
        # Desktop
        browser.page.set_viewport_size({"width": 1920, "height": 1080})
        browser.navigate("http://localhost:3000")
        assert browser.element_exists(".container")

        # Mobile
        browser.page.set_viewport_size({"width": 375, "height": 667})
        assert browser.element_exists(".container")
        print("✅ Responsive layout")

def test_5_performance():
    """Test page load performance"""
    import time
    with PlaywrightAutomation() as browser:
        start = time.time()
        browser.navigate("http://localhost:3000")
        browser.page.wait_for_selector("canvas")
        load_time = time.time() - start

        assert load_time < 5, f"Load time {load_time:.2f}s too slow"
        print(f"✅ Performance ({load_time:.2f}s)")

# Run all tests
if __name__ == "__main__":
    tests = [
        test_1_component_loads,
        test_2_data_fetches,
        test_3_filters_work,
        test_4_responsive,
        test_5_performance
    ]

    for test in tests:
        test()

    print("\n" + "="*60)
    print("🎉 All automated tests passed!")
    print("="*60)
    print("\n⚠️  Manual verification still required:")
    print("  [ ] Globe renders correctly")
    print("  [ ] Markers in right positions")
    print("  [ ] Animations smooth")
    print("\n💡 Run: ./scripts/visual-verify-globe.sh")
```

## CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  automated:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: npm install
      - name: Start dev server
        run: npm run dev &
      - name: Wait for server
        run: sleep 5
      - name: Run WebPilot tests
        run: python tests/webpilot-globe-test.py

      - name: Comment on PR
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              body: '✅ Automated tests passed!\n\n⚠️ Manual verification required:\n- [ ] Globe renders\n- [ ] Markers visible\n- [ ] Animations smooth'
            })
```

## Summary

**Accept the limitation**: WebGL can't be screenshot-tested automatically

**Embrace the hybrid**:
- ✅ 90% automated (data, logic, interactions)
- ⚠️ 10% manual (visual appearance)
- ⏱️ 30 seconds automated + 2 minutes manual = 2.5 minutes total

**Document everything**:
- Screenshots for visual changes
- Test results for functional changes
- Both in PRs for review

This is **realistic, sustainable, and catches bugs effectively**.

---

## Files to Create

When setting up Terra Atlas, create these files:

```
terra-atlas-app/
├── tests/
│   ├── README.md (testing guide)
│   └── webpilot-globe-test.py (automated tests)
├── scripts/
│   └── visual-verify-globe.sh (manual helper)
├── docs/
│   ├── VISUAL_VERIFICATION_STRATEGY.md (full strategy)
│   └── visual-evidence/ (screenshots directory)
└── package.json (add test scripts)
```

All template code is provided above. Copy-paste and customize as needed.
