# Licensing FAQ — Luminous Dynamics

Note: some vendored/third-party components and git submodules may carry their own
licenses; see the `LICENSE` files in those directories.

## For Researchers and Students

**Q: Can I use Symthaea/Mycelix for my research?**
A: Yes. Academic research, education, and personal experimentation are fully
permitted under the AGPL-3.0. No commercial license is needed. Cite us if
you publish.

**Q: Can I use the psych-bench suite to benchmark my own AI system?**
A: Yes. The benchmarks are AGPL-licensed. If you modify the benchmark suite
itself and distribute it, share your modifications under the AGPL.

**Q: Can I use this codebase for my thesis or coursework?**
A: Yes. No restrictions on academic use.

## For Developers

**Q: Can I fork this and build my own project on top of it?**
A: Yes, as long as your project is also licensed under the AGPL-3.0-or-later
and you make your source code available. The AGPL's copyleft applies to
derivative works.

**Q: What if I only use one crate (e.g., symthaea-core)?**
A: The AGPL applies to the crate and any work that links against it. If you
distribute a binary or run a network service that includes an AGPL-licensed
crate, the entire combined work must be available under the AGPL.

**Q: Can I contribute a patch or feature?**
A: Yes. By submitting a pull request, you agree to our Contributor License
Agreement (see CLA.md). You retain your copyright; you grant Luminous
Dynamics the right to use your contribution under any license, including
commercial licenses. Your contribution will always remain available to the
public under the AGPL.

**Q: I contributed code. Can I still use my own code however I want?**
A: Yes. The CLA is a license grant, not a copyright assignment. You retain
full ownership of your contribution and can use it in any other project
under any license.

## For Companies

**Q: Can I use this software internally without a commercial license?**
A: Internal use (not distributed, not offered as a network service) is
permitted under the AGPL. If you modify the software and only use it
internally without making it available to third parties, no commercial
license is needed.

**Q: Can I run a modified version as a SaaS product?**
A: Under the AGPL, yes — but you must make the complete source code of your
modified version available to all users of the service. If you cannot or
will not do this, you need a commercial license from Luminous Dynamics.

**Q: Can I embed this in a proprietary product I sell to customers?**
A: Not under the AGPL. The AGPL requires that the source code of the entire
combined work be available to recipients. If your product is proprietary,
you need a commercial license.

**Q: How much does a commercial license cost?**
A: Pricing is negotiated based on scope of use. We offer favorable terms
for cooperatives, public-benefit corporations, and mission-aligned
organizations. Contact tristan.stoltz@evolvingresonantcocreationism.com.

**Q: Can I evaluate the software before purchasing a commercial license?**
A: Yes. Evaluation and testing are always free. A commercial license is only
required when you deploy in production in a way that conflicts with the AGPL.

## For Open-Source Projects

**Q: Can I use Luminous Dynamics code in my GPL-3.0 project?**
A: The AGPL-3.0 is compatible with GPL-3.0 in one direction: AGPL code can
be combined with GPL-3.0 code, but the combined work must be distributed
under the AGPL-3.0. See Section 13 of the AGPL for details.

**Q: Is the AGPL compatible with MIT/Apache-2.0?**
A: You can include MIT/Apache-2.0 code in an AGPL project. You cannot
include AGPL code in a MIT/Apache-2.0 project without the entire combined
work becoming AGPL-licensed.

**Q: What about the LGPL? Can I link against your libraries without the
copyleft applying to my code?**
A: No. The AGPL does not have an LGPL-style linking exception. Any work
that links against AGPL-licensed code is a derivative work and must be
licensed under the AGPL. If this is a problem for your project, contact us
about a commercial license.

## About the License Change

**Q: Wasn't this repository previously BSL 1.1?**
A: Yes. The root license was previously the Business Source License 1.1,
which would have converted to Apache-2.0 on 2029-03-06. The legacy text is
archived at `LICENSE-BSL-1.1.txt`. We replaced the repo-root license with
the AGPL-3.0-or-later because:
1. The AGPL provides stronger copyleft protection (especially the network
   interaction clause in Section 13).
2. The AGPL is a true open-source license recognized by the OSI and FSF.
3. The BSL's time-bomb conversion to Apache-2.0 would have eventually
   removed all copyleft protection, allowing extractive use without
   contributing back.

**Q: Symthaea was already AGPL. What changed?**
A: The main Symthaea crate was already AGPL-3.0-or-later. This change
aligns the repo-root license with Symthaea and reconciles first-party
sub-crates/SDK metadata that previously reported MIT or Apache-2.0.

---

*For questions not covered here: tristan.stoltz@evolvingresonantcocreationism.com*
