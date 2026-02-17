import { Router } from 'express';
import Joi from 'joi';
import { accountController } from '../controllers/account.controller';
import { authenticate } from '../middleware/auth';
import { validate } from '../middleware/validate';

const router = Router();

const createAccountSchema = Joi.object({
  email: Joi.string().email().required(),
  provider: Joi.string().required(),
  imapHost: Joi.string().required(),
  imapPort: Joi.number().required(),
  imapSecure: Joi.boolean().default(true),
  imapUser: Joi.string().required(),
  imapPassword: Joi.string().required(),
  smtpHost: Joi.string().required(),
  smtpPort: Joi.number().required(),
  smtpSecure: Joi.boolean().default(true),
  smtpUser: Joi.string().required(),
  smtpPassword: Joi.string().required(),
});

router.use(authenticate);

router.get('/', accountController.getAll);
router.post('/', validate(createAccountSchema), accountController.create);
router.get('/:id', accountController.getOne);
router.put('/:id', accountController.update);
router.delete('/:id', accountController.delete);
router.put('/:id/default', accountController.setDefault);

export { router as accountRoutes };
