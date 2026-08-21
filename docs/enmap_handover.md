# Downloading EnMAP L2A — instructions for a helper

Everything here was checked against the live sites on **2026-08-21**. If a step does not match
what you see, stop and say so rather than improvising — DLR's own documentation is out of date in
several places and this project has been caught by that repeatedly.

---

## 0. First, the naming confusion

Searching "EnMAP access service" lands on the wrong site. **Five** DLR properties carry the EnMAP
name and only one of them serves the data we need:

| site | what it is | use it? |
|---|---|---|
| `enmap.org` | mission website — documentation, news, FAQs. **No data.** | no |
| `planning.enmap.org` | Instrument Planning Portal (tasking requests). **Refuses connections as of 2026-08-21.** | no |
| `eoweb.dlr.de` | EOWEB GeoPortal — cart/order UI with on-demand reprocessing | fallback only |
| **`geoservice.dlr.de`** | **EOC Geoservice — direct download of pre-processed L2A** | **yes** |
| `sso.eoc.dlr.de` | the login server for Geoservice | yes (you get sent here) |

Everything below stays inside **EOC Geoservice**.

One more wrinkle worth knowing before it confuses you: `sso.eoc.dlr.de` runs **two** login systems.
Permissions are managed in **Keycloak** (`/eoc/kc/realms/geoservice/…`); the download wall is
**CAS** (`/eoc/auth/login`). Being logged into one does not mean you are authorised in the other.

---

## 1. Register

<https://sso.eoc.dlr.de/geoservice/selfservice/public/newuser?locale=en>

Enter your email, confirm via the emailed link, set a password.

> **Write down the username.** It is **not** your email address — the system assigns or asks for a
> separate one. Not knowing this cost us an hour.

---

## 2. Subscribe to the EnMAP Access Service

<https://sso.eoc.dlr.de/eoc/kc/realms/geoservice/account/#/permissions>

Under **"Permissions available for subscription"**, find **EnMAP Access Service**, click **+**, and
accept the licence terms in the dialogue that appears. It should then move up into **"Permissions
you are subscribed to."**

---

## 3. Close every DLR tab

Not optional. The download service rejects sessions with this message:

> *"Service access denied due to missing privileges. You may see this notification because you are
> actually logged into another of our services with another account."*

Having the permissions page open in another tab is enough to trigger it. Close everything, or use
a fresh private window that you do **not** use for step 2.

---

## 4. Log in at the download service itself

<https://download.geoservice.dlr.de/ENMAP/files/L2A/>

You will be redirected to a CAS login page titled **"EOC UMS: EnMAP Access Service"**. Log in with
the **username from step 1** (not the email).

If that page shows *Authentication Failure*, see §7.

---

## 5. Download the files

`docs/enmap_download_list.txt` holds **40 URLs across 8 scenes** — five files per scene:

| file | why we need it |
|---|---|
| `…-SPECTRAL_IMAGE_COG.TIF` | the 224-band cube. **The important one.** ~1 GB each |
| `…-METADATA.XML` | carries the wavelength table — **do not skip this** |
| `…-QL_QUALITY_CLASSES_COG.TIF` | cloud/shadow masking |
| `…-QL_QUALITY_CLOUD_COG.TIF` | cloud mask |
| `…-QL_PIXELMASK_COG.TIF` | defective pixels |

Paste each URL into the browser tab where you logged in at step 4.

**Please download scene 1 first and confirm it opens** before doing the other seven — that is
~8 GB total and there is no point discovering a problem at the end.

Do **not** download the `QL_VNIR`, `QL_SWIR` or `_thumbnail.jpg` files. They are browse overviews,
not science data, and they roughly triple the transfer for nothing.

---

## 6. Sending the files back

Keep the **original filenames** and the per-scene folder structure. Filenames encode the datatake
and processing version, and the pipeline's manifest check matches on them.

If you can, run `sha256sum` over each file and send the checksums alongside — we verify every file
against a manifest before it enters the pipeline, and a silently truncated 1 GB download is
otherwise very hard to spot.

---

## 7. If it still fails

The error we hit repeatedly is *Authentication Failure — service access denied due to missing
privileges*, even with the subscription showing as active.

1. Confirm the subscription really is under **"Permissions you are subscribed to"**, not
   "available for subscription".
2. Try the **EO-Lab (ENMAP)** button on the right of the login page — a separate identity provider.
3. Try downloading anything from **DESIS** instead. If DESIS works and EnMAP does not, the fault
   is EnMAP-specific rather than account-wide, which is exactly what the helpdesk needs to hear.
4. Contact **`eoc-ums-helpdesk@dlr.de`** — the address the login page itself gives for
   user-management problems. (Not `erdbeobachtung@dlr.de`; that handles the 5 000-product
   contingent, a different question.)

---

## 8. Notes

- EnMAP users may download **up to 5 000 L2A products**. Eight is nowhere near the limit.
- Check the licence on <https://geoservice.dlr.de/web/datasets/enmap_l2_hsi> before passing files
  on. It is open research data, but confirm rather than assume.
- **Fallback route:** the EOWEB GeoPortal (<https://eoweb.dlr.de/egp/>) offers the same archive
  through a cart-and-order UI with on-demand reprocessing. Slower, but a different code path — if
  Geoservice stays blocked, it is worth a try.
