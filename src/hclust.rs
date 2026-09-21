//! `hclust` and `cutree`, transliterated from R 4.0.4's `hclust.f` and `hclust-utils.c`.
//!
//! ConsensusClusterPlus calls both a hundred times per run, and the answer FlowSOM's
//! metaclustering gives is the one they produce — ties and all. Which pair a tie resolves to is
//! decided by the scan order in the Fortran, so the loops here keep it: strict `<`, first match
//! wins, indices scanned ascending.
//!
//! Distances are held the way the Fortran holds them, in lower-half-diagonal order, so `IOFFST`
//! below is the original's index arithmetic rather than a translation of it.

/// `IOFFST(N,I,J)` for 1-based `i < j`, returning a 0-based offset.
#[inline]
fn ioffst(n: usize, i: usize, j: usize) -> usize {
    j + (i - 1) * n - (i * (i + 1)) / 2 - 1
}

/// Euclidean `dist()` over the rows of a column-major matrix, in R's lower-triangle order.
pub fn dist_euclidean(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut d = Vec::with_capacity(n * (n - 1) / 2);
    for i in 1..=n {
        for j in i + 1..=n {
            let mut acc = 0.0;
            for k in 0..p {
                let t = x[(i - 1) + k * n] - x[(j - 1) + k * n];
                acc += t * t;
            }
            d.push(acc.sqrt());
        }
    }
    // R fills column by column of the lower triangle; the loop above fills row by row, so
    // reorder into the same layout IOFFST assumes.
    let mut out = vec![0.0; d.len()];
    let mut at = 0;
    for i in 1..=n {
        for j in i + 1..=n {
            out[ioffst(n, i, j)] = d[at];
            at += 1;
        }
    }
    out
}

/// Linkage criteria, numbered as R's `hclust` numbers them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    Single = 2,
    Complete = 3,
    /// `"average"`, the only one FlowSOM and ConsensusClusterPlus use.
    Average = 4,
    McQuitty = 5,
}

/// What R's `hclust` returns: `merge` in R's own encoding (negative = singleton, positive =
/// an earlier row), the heights, and the dendrogram order.
#[derive(Debug, Clone)]
pub struct Hclust {
    pub merge: Vec<(i32, i32)>,
    pub height: Vec<f64>,
    pub order: Vec<usize>,
}

const INF: f64 = 1.0e300;

/// `HCLUST` + `HCASS2`. `diss` is consumed: the Fortran updates it in place.
pub fn hclust(mut diss: Vec<f64>, n: usize, linkage: Linkage) -> Hclust {
    assert_eq!(diss.len(), n * (n - 1) / 2, "lower-triangle length");
    let mut membr = vec![1.0f64; n + 1];
    let mut flag = vec![true; n + 1];
    let mut nn = vec![0usize; n + 1];
    let mut disnn = vec![0.0f64; n + 1];
    let mut ia = vec![0usize; n + 1];
    let mut ib = vec![0usize; n + 1];
    let mut crit = vec![0.0f64; n + 1];

    // Nearest neighbour to the RIGHT of each point.
    for i in 1..n {
        let mut dmin = INF;
        let mut jm = 0;
        for j in i + 1..=n {
            let d = diss[ioffst(n, i, j)];
            if dmin > d {
                dmin = d;
                jm = j;
            }
        }
        nn[i] = jm;
        disnn[i] = dmin;
    }

    let mut ncl = n;
    let mut jj = 0usize;
    loop {
        // The least dissimilarity among the nearest-neighbour list.
        let mut dmin = INF;
        let (mut im, mut jm) = (0usize, 0usize);
        for i in 1..n {
            if flag[i] && disnn[i] < dmin {
                dmin = disnn[i];
                im = i;
                jm = nn[i];
            }
        }
        ncl -= 1;

        let i2 = im.min(jm);
        let j2 = im.max(jm);
        ia[n - ncl] = i2;
        ib[n - ncl] = j2;
        crit[n - ncl] = dmin;
        flag[j2] = false;

        // Update dissimilarities from the new cluster.
        let mut dmin = INF;
        for k in 1..=n {
            if !flag[k] || k == i2 {
                continue;
            }
            let ind1 = if i2 < k {
                ioffst(n, i2, k)
            } else {
                ioffst(n, k, i2)
            };
            let ind2 = if j2 < k {
                ioffst(n, j2, k)
            } else {
                ioffst(n, k, j2)
            };
            diss[ind1] = match linkage {
                Linkage::Single => diss[ind1].min(diss[ind2]),
                Linkage::Complete => diss[ind1].max(diss[ind2]),
                Linkage::Average => {
                    (membr[i2] * diss[ind1] + membr[j2] * diss[ind2]) / (membr[i2] + membr[j2])
                }
                Linkage::McQuitty => (diss[ind1] + diss[ind2]) / 2.0,
            };
            if i2 < k {
                if diss[ind1] < dmin {
                    dmin = diss[ind1];
                    jj = k;
                }
            } else if diss[ind1] < disnn[k] {
                // The fix that keeps nearest neighbours correct for non-monotone criteria.
                disnn[k] = diss[ind1];
                nn[k] = i2;
            }
        }
        membr[i2] += membr[j2];
        disnn[i2] = dmin;
        nn[i2] = jj;

        // Redetermine the nearest neighbour of anything that pointed at the merged pair.
        for i in 1..n {
            if flag[i] && (nn[i] == i2 || nn[i] == j2) {
                let mut dmin = INF;
                for j in i + 1..=n {
                    if !flag[j] {
                        continue;
                    }
                    let d = diss[ioffst(n, i, j)];
                    if d < dmin {
                        dmin = d;
                        jj = j;
                    }
                }
                nn[i] = jj;
                disnn[i] = dmin;
            }
        }

        if ncl <= 1 {
            break;
        }
    }

    let (merge, order) = hcass2(n, &ia, &ib);
    Hclust {
        merge,
        height: crit[1..n].to_vec(),
        order,
    }
}

/// `HCASS2`: recode the agglomerations the way `plclust` wants them, and work out the
/// dendrogram order.
fn hcass2(n: usize, ia: &[usize], ib: &[usize]) -> (Vec<(i32, i32)>, Vec<usize>) {
    let mut iia: Vec<i32> = (0..=n)
        .map(|i| if i == 0 { 0 } else { ia[i] as i32 })
        .collect();
    let mut iib: Vec<i32> = (0..=n)
        .map(|i| if i == 0 { 0 } else { ib[i] as i32 })
        .collect();

    for i in 1..=n.saturating_sub(2) {
        let k = ia[i].min(ib[i]) as i32;
        for j in i + 1..n {
            if ia[j] as i32 == k {
                iia[j] = -(i as i32);
            }
            if ib[j] as i32 == k {
                iib[j] = -(i as i32);
            }
        }
    }
    for i in 1..n {
        iia[i] = -iia[i];
        iib[i] = -iib[i];
        if iia[i] > 0 && iib[i] < 0 {
            let k = iia[i];
            iia[i] = iib[i];
            iib[i] = k;
        }
        if iia[i] > 0 && iib[i] > 0 {
            let (k1, k2) = (iia[i].min(iib[i]), iia[i].max(iib[i]));
            iia[i] = k1;
            iib[i] = k2;
        }
    }

    // The dendrogram order: start from the last merge and expand each positive entry in place.
    let mut iorder = vec![0i32; n + 1];
    iorder[1] = iia[n - 1];
    iorder[2] = iib[n - 1];
    let mut loc = 2usize;
    for i in (1..=n - 2).rev() {
        for j in 1..=loc {
            if iorder[j] == i as i32 {
                iorder[j] = iia[i];
                if j == loc {
                    loc += 1;
                    iorder[loc] = iib[i];
                } else {
                    loc += 1;
                    let mut k = loc;
                    while k >= j + 2 {
                        iorder[k] = iorder[k - 1];
                        k -= 1;
                    }
                    iorder[j + 1] = iib[i];
                }
                break;
            }
        }
    }

    let merge = (1..n).map(|i| (iia[i], iib[i])).collect();
    let order = (1..=n).map(|i| (-iorder[i]) as usize).collect();
    (merge, order)
}

/// `cutree(tree, k)`: cluster numbers in observation order, 1-based, numbered by first
/// appearance — which is why the result cannot be derived from the merge matrix alone without
/// walking it in exactly this order.
pub fn cutree(merge: &[(i32, i32)], k: usize) -> Vec<usize> {
    let n = merge.len() + 1;
    if k == n {
        return (1..=n).collect();
    }
    let mut sing = vec![true; n + 1];
    let mut m_nr = vec![0usize; n + 1];
    let mut ans = vec![0usize; n];

    for step in 1..=n - 1 {
        let (mut m1, m2) = merge[step - 1];
        if m1 < 0 && m2 < 0 {
            m_nr[(-m1) as usize] = step;
            m_nr[(-m2) as usize] = step;
            sing[(-m1) as usize] = false;
            sing[(-m2) as usize] = false;
        } else if m1 < 0 || m2 < 0 {
            let j = if m1 < 0 {
                let j = -m1;
                m1 = m2;
                j
            } else {
                -m2
            } as usize;
            for l in 1..=n {
                if m_nr[l] == m1 as usize {
                    m_nr[l] = step;
                }
            }
            m_nr[j] = step;
            sing[j] = false;
        } else {
            for l in 1..=n {
                if m_nr[l] == m1 as usize || m_nr[l] == m2 as usize {
                    m_nr[l] = step;
                }
            }
        }

        if k == n - step {
            let mut z = vec![0usize; n + 1];
            let mut nclust = 0;
            for l in 1..=n {
                if sing[l] {
                    nclust += 1;
                    ans[l - 1] = nclust;
                } else {
                    if z[m_nr[l]] == 0 {
                        nclust += 1;
                        z[m_nr[l]] = nclust;
                    }
                    ans[l - 1] = z[m_nr[l]];
                }
            }
        }
    }
    ans
}
