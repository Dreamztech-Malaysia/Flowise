import express from 'express'
import customerMemoryController from '../../controllers/customer-memory'
import { checkAnyPermission } from '../../enterprise/rbac/PermissionCheck'

const router = express.Router()

const VIEW = 'chatflows:view,chatflows:update,agentflows:view,agentflows:update'
const DELETE = 'chatflows:delete,agentflows:delete'

// READ
router.get('/', checkAnyPermission(VIEW), customerMemoryController.listUsers)
// must stay above '/:userId' so 'summary' is not taken as a user id
router.get('/summary', checkAnyPermission(VIEW), customerMemoryController.getSummary)
router.get('/:userId', checkAnyPermission(VIEW), customerMemoryController.getFacts)

// DELETE
router.delete('/:userId', checkAnyPermission(DELETE), customerMemoryController.deleteAllFacts)
router.delete('/:userId/:memoryId', checkAnyPermission(DELETE), customerMemoryController.deleteFact)

export default router
