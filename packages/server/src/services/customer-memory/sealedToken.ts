import { createCipheriv, createDecipheriv, createHash, randomBytes } from 'crypto'
import { StatusCodes } from 'http-status-codes'
import { InternalFlowiseError } from '../../errors/internalFlowiseError'

/**
 * Sealed customer-memory token (agentflows).
 *
 * whatsapp.dt seals the overrideConfig.customerMemory block (it carries the customer's own LLM key) with
 * AES-256-GCM under CUSTOMER_MEMORY_TOKEN_SECRET and passes the token to the flow as a start-state value. The flow's
 * Custom Function nodes post it back to /api/v1/customer-memory/sealed/*. So:
 *   - the key never appears in plain text in the flow, its state or its execution logs;
 *   - the token is the auth: only the holder of the secret can mint one, it names exactly one customer, and it expires.
 *
 * Format: `cm1.` + base64url( iv(12) | ciphertext | tag(16) ), plaintext JSON `{ "exp": <unix s>, "cm": {...} }`.
 * Key = sha256(secret). PHP: openssl_encrypt($json, 'aes-256-gcm', hash('sha256', $secret, true), OPENSSL_RAW_DATA, $iv, $tag).
 */

export const SEALED_PREFIX = 'cm1.'
/** Upper bound on token lifetime regardless of what the minting side asks for. */
export const SEALED_MAX_TTL_S = 24 * 3600

const key = (): Buffer => {
    const secret = process.env.CUSTOMER_MEMORY_TOKEN_SECRET
    if (!secret || secret.length < 16) {
        throw new InternalFlowiseError(
            StatusCodes.SERVICE_UNAVAILABLE,
            'Sealed customer memory is not configured (CUSTOMER_MEMORY_TOKEN_SECRET)'
        )
    }
    return createHash('sha256').update(secret).digest()
}

const b64url = (b: Buffer) => b.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
const unb64url = (s: string) => Buffer.from(s.replace(/-/g, '+').replace(/_/g, '/'), 'base64')

/** Mint a token (tests and tooling; production tokens are minted by whatsapp.dt). */
export const sealCustomerMemory = (cm: Record<string, any>, ttlSeconds = 900, now = Date.now()): string => {
    const iv = randomBytes(12)
    const cipher = createCipheriv('aes-256-gcm', key(), iv)
    const plain = Buffer.from(JSON.stringify({ exp: Math.floor(now / 1000) + ttlSeconds, cm }), 'utf8')
    const enc = Buffer.concat([cipher.update(plain), cipher.final()])
    return SEALED_PREFIX + b64url(Buffer.concat([iv, enc, cipher.getAuthTag()]))
}

/** Open a token. Any problem (format, tamper, wrong secret, expiry) => 401, never the reason in detail. */
export const openCustomerMemory = (token: unknown, now = Date.now()): Record<string, any> => {
    const deny = () => new InternalFlowiseError(StatusCodes.UNAUTHORIZED, 'Invalid or expired customer memory token')
    if (typeof token !== 'string' || !token.startsWith(SEALED_PREFIX) || token.length > 8192) throw deny()
    const k = key()
    const buf = unb64url(token.slice(SEALED_PREFIX.length))
    if (buf.length < 12 + 16 + 2) throw deny()
    let parsed: any
    try {
        const decipher = createDecipheriv('aes-256-gcm', k, buf.subarray(0, 12))
        decipher.setAuthTag(buf.subarray(buf.length - 16))
        const plain = Buffer.concat([decipher.update(buf.subarray(12, buf.length - 16)), decipher.final()]).toString('utf8')
        parsed = JSON.parse(plain)
    } catch {
        throw deny()
    }
    const nowS = Math.floor(now / 1000)
    const exp = Number(parsed?.exp)
    if (!Number.isFinite(exp) || exp < nowS || exp > nowS + SEALED_MAX_TTL_S) throw deny()
    if (!parsed.cm || typeof parsed.cm !== 'object' || Array.isArray(parsed.cm)) throw deny()
    return parsed.cm
}
